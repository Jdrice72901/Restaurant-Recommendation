import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import bm25s
import chromadb
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from recommender.index_baseline import chunks
from recommender.prep_baseline import write_json


def load_settings(root, path=None):
    path = Path(path) if path else root / 'config_project.yaml'
    if not path.is_absolute():
        path = root / path
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    evidence, retrieval = cfg['evidence'], cfg['retrieval']
    if not 0 <= evidence['overlap_tokens'] < evidence['max_tokens']:
        raise ValueError("Invalid passage token budget/overlap")
    if retrieval['fusion'] != 'reciprocal_rank' or retrieval['reranker'] is not None:
        raise ValueError("This builder implements BM25/Gemma RRF without a reranker")
    for key in ['fusion_k', 'per_retriever_restaurants', 'candidate_restaurants']:
        if type(retrieval[key]) is not int or retrieval[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return cfg


class GemmaEmbedder:
    def __init__(self, cfg, device=None, batch_size=32):
        from sentence_transformers import SentenceTransformer

        if batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")
        self.batch_size = batch_size
        self.cfg = cfg['retrieval']
        self.model = SentenceTransformer(
            self.cfg['embedding_model'], revision=self.cfg['revision'], device=device
        )
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.max_seq_length

    def encode(self, texts, query=False):
        prompt = self.cfg['query_prompt' if query else 'document_prompt']
        return self.model.encode(
            texts,
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


def source_chunks(text, embedder, cfg):
    """Count the document prompt in the budget, keeps unprefixed source text."""
    tokenizer = embedder.tokenizer
    prompt = cfg['retrieval']['document_prompt']
    maximum = cfg['evidence']['max_tokens']
    if maximum > embedder.max_seq_length:
        raise ValueError("Passage budget exceeds model context")
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    overlap = cfg['evidence']['overlap_tokens']
    # Tokenization at the prompt/text boundary can change
    budget = maximum - prompt_tokens
    while True:
        result = list(chunks(text, tokenizer, budget, overlap))
        if all(
            len(tokenizer.encode(prompt + part, add_special_tokens=True)) <= maximum
            for part, _, _ in result
        ):
            return result
        budget -= 1


def build(root, cfg, embedder, mode='evaluation', limit=None, batch_size=64):
    """Publish only after both stores are complete, trials use separate pointers."""
    if mode not in {'evaluation', 'serving'}:
        raise ValueError("mode must be evaluation or serving")
    if batch_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("batch_size and trial limit must be positive")
    started = perf_counter()
    active = json.loads((root / 'data/processed/analytical/active.json').read_text())
    prepared = root / 'data/processed/philadelphia' / active['run']
    readiness = json.loads((prepared / 'readiness.json').read_text())
    if readiness['decisions'] != cfg:
        raise ValueError("Project settings changed; rerun 01_data_readiness.py first")
    index_root = root / 'index/hybrid' / mode
    run = index_root / 'builds' / uuid4().hex
    run.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(run / 'chroma'))
    collection = client.create_collection(
        'passages', embedding_function=None, metadata={'hnsw:space': 'cosine'}
    )
    source = root / readiness['review_source']
    passage_schema = pa.schema(
        [
            ('passage_id', pa.string()),
            ('review_id', pa.string()),
            ('business_id', pa.string()),
            ('date', pa.string()),
            ('review_stars', pa.float64()),
            ('source_start', pa.int64()),
            ('source_end', pa.int64()),
            ('text', pa.string()),
        ]
    )
    pending, total, review_count = [], 0, 0

    def flush(writer):
        nonlocal total
        collection.add(
            ids=[p['passage_id'] for p in pending],
            documents=[p['text'] for p in pending],
            metadatas=[
                {'business_id': p['business_id'], 'review_id': p['review_id']} for p in pending
            ],
            embeddings=embedder.encode([p['text'] for p in pending]),
        )
        writer.write_table(pa.Table.from_pylist(pending, schema=passage_schema))
        total += len(pending)
        pending.clear()
        print(f'Indexed {total:,} passages', flush=True)

    with duckdb.connect() as con:
        con.read_parquet(str(source)).create_view('source_reviews')
        con.read_parquet(str(prepared / 'index_membership.parquet')).create_view('membership')
        eligible = 'evaluation_evidence' if mode == 'evaluation' else 'serving_evidence'
        selection = f"""SELECT r.* FROM source_reviews r SEMI JOIN
            (SELECT review_id FROM membership WHERE {eligible}) m USING(review_id)
            WHERE r.text IS NOT NULL AND length(trim(r.text))>0"""
        if limit is not None:
            selection += f" ORDER BY md5(review_id || '{cfg['retrieval']['seed']}'),review_id LIMIT {int(limit)}"
        con.execute("CREATE TEMP TABLE evidence AS " + selection)
        con.read_parquet(str(prepared / 'restaurants.parquet')).create_view('catalog')
        catalog = con.sql("SELECT * FROM catalog SEMI JOIN evidence USING(business_id)").df()
        catalog.to_parquet(run / 'restaurants.parquet', index=False)
        cursor = con.execute(
            "SELECT review_id,business_id,date,stars,text FROM evidence ORDER BY review_id"
        )
        with pq.ParquetWriter(
            run / 'passages.parquet', passage_schema, compression='zstd'
        ) as writer:
            while records := cursor.fetchmany(batch_size):
                for review_id, business_id, date, rating, text in records:
                    review_count += 1
                    for number, (part, begin, end) in enumerate(source_chunks(text, embedder, cfg)):
                        pending.append(
                            {
                                'passage_id': f'{review_id}:{number}',
                                'review_id': review_id,
                                'business_id': business_id,
                                'date': date.isoformat(),
                                'review_stars': float(rating),
                                'source_start': begin,
                                'source_end': end,
                                'text': part,
                            }
                        )
                        if len(pending) == batch_size:
                            flush(writer)
            if pending:
                flush(writer)
    if not total or collection.count() != total:
        raise ValueError("Empty or incomplete dense corpus")
    passages = pd.read_parquet(run / 'passages.parquet')
    assert passages.passage_id.is_unique
    tokens = bm25s.tokenize(passages.text.tolist(), stopwords=[], show_progress=False)
    lexical = bm25s.BM25(
        method='lucene', k1=cfg['retrieval']['bm25_k1'], b=cfg['retrieval']['bm25_b']
    )
    lexical.index(tokens, show_progress=False)
    lexical.save(str(run / 'bm25'), show_progress=False)
    # Verify the persisted lexical matrix addresses exactly the shared passage rows
    reloaded = bm25s.BM25.load(str(run / 'bm25'), mmap=True, show_progress=False)
    if len(reloaded.get_scores(['restaurant'])) != total:
        raise ValueError("Lexical and dense passage counts differ")
    manifest = {
        'version': 1,
        'run': run.name,
        'analytical_run': active['run'],
        'settings': cfg,
        'mode': mode,
        'trial': limit is not None,
        'limit': limit,
        'reviews': review_count,
        'restaurants': len(catalog),
        'passages': total,
        'build_seconds': perf_counter() - started,
        'artifact_bytes': sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),
    }
    write_json(run / 'manifest.json', manifest)
    write_json(index_root / ('trial.json' if limit is not None else 'active.json'), manifest)
    return manifest


class HybridIndex:
    """Load a frozen index and apply identical restaurant eligibility to both paths."""

    def __init__(self, root, cfg, embedder, mode='serving', trial=False):
        index_root = root / 'index/hybrid' / mode
        self.manifest = json.loads(
            (index_root / ('trial.json' if trial else 'active.json')).read_text()
        )
        if self.manifest['settings'] != cfg or self.manifest['mode'] != mode:
            raise ValueError("Index settings changed; rebuild before querying")
        run = index_root / 'builds' / self.manifest['run']
        self.cfg, self.embedder = cfg, embedder
        self.catalog = pd.read_parquet(run / 'restaurants.parquet')
        self.passages = pd.read_parquet(run / 'passages.parquet')
        self.lexical = bm25s.BM25.load(str(run / 'bm25'), mmap=True, show_progress=False)
        self.collection = chromadb.PersistentClient(path=str(run / 'chroma')).get_collection(
            'passages', embedding_function=None
        )
        if (
            self.collection.count() != len(self.passages)
            or len(self.passages) != self.manifest['passages']
        ):
            raise ValueError("Stored passage counts do not match manifest")

    def search(self, query, max_price=None, delivery=False, takeout=False, outdoor=False):
        if not query.strip():
            return []
        # Refuse silent query truncation as well as document truncation
        prompt = self.cfg['retrieval']['query_prompt']
        if (
            len(self.embedder.tokenizer.encode(prompt + query, add_special_tokens=True))
            > self.embedder.max_seq_length
        ):
            raise ValueError("Query exceeds embedding model context")
        eligible = self.catalog.copy()
        if max_price is not None:
            if max_price not in [1, 2, 3, 4]:
                raise ValueError("max_price must be in 1..4")
            eligible = eligible.loc[eligible.price.notna() & eligible.price.le(max_price)]
        for required, column in [
            (delivery, 'delivery'),
            (takeout, 'takeout'),
            (outdoor, 'outdoor_seating'),
        ]:
            if required:
                eligible = eligible.loc[eligible[column].eq(True).fillna(False)]
        bids = set(eligible.business_id)
        if not bids:
            return []
        depth = self.cfg['retrieval']['per_retriever_restaurants']
        allowed = self.passages.business_id.isin(bids).to_numpy()
        query_tokens = bm25s.tokenize([query], stopwords=[], return_ids=False, show_progress=False)[
            0
        ]
        scores = self.lexical.get_scores(query_tokens) if query_tokens else np.zeros(len(allowed))
        lexical = self.passages.loc[allowed & (scores > 0)].assign(
            score=scores[allowed & (scores > 0)]
        )
        lexical = (
            lexical.sort_values(['score', 'passage_id'], ascending=[False, True])
            .drop_duplicates('business_id')
            .head(depth)
        )
        where = {'business_id': {'$in': sorted(bids)}}
        count = int(allowed.sum())
        n = min(count, max(depth * 4, 1))
        vector = self.embedder.encode([query], query=True)
        while True:
            response = self.collection.query(
                query_embeddings=vector,
                n_results=n,
                where=where,
                include=['metadatas', 'distances'],
            )
            dense_rows = []
            seen = set()
            for pid, meta, distance in zip(
                response['ids'][0], response['metadatas'][0], response['distances'][0], strict=True
            ):
                if meta['business_id'] not in seen:
                    seen.add(meta['business_id'])
                    dense_rows.append(
                        {
                            'business_id': meta['business_id'],
                            'passage_id': pid,
                            'score': 1 - distance,
                        }
                    )
            if len(dense_rows) >= min(depth, len(bids)) or n == count:
                break
            n = min(count, n * 2)
        sources = {}
        for name, records in [('bm25', lexical.to_dict('records')), ('gemma', dense_rows[:depth])]:
            for rank, record in enumerate(records, 1):
                bid = record['business_id']
                item = sources.setdefault(bid, {'business_id': bid, 'score': 0.0, 'evidence': []})
                item['score'] += 1 / (self.cfg['retrieval']['fusion_k'] + rank)
                item['evidence'].append(
                    {'retriever': name, 'rank': rank, 'passage_id': record['passage_id']}
                )
        results = sorted(sources.values(), key=lambda r: (-r['score'], r['business_id']))[
            : self.cfg['retrieval']['candidate_restaurants']
        ]
        lookup = self.passages.set_index('passage_id')
        for item in results:
            for evidence in item['evidence']:
                evidence.update(lookup.loc[evidence['passage_id']].to_dict())
        return results
