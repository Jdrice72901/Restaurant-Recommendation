import json
import re
from uuid import uuid4

import chromadb
import duckdb

from recommender.prep_baseline import select_evidence, sql_string, write_json


class Embedder:
    def __init__(self, cfg):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(cfg['embedding_model'], revision=cfg['revision'])
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.max_seq_length

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def chunks(text, tokenizer, max_tokens, overlap):
    """Keep exact source substrings and prefer sentence ends within token windows."""
    encoded = tokenizer(
        text, add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False
    )
    offsets = encoded['offset_mapping']
    budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= overlap or overlap < 0:
        raise ValueError("Chunk overlap must be smaller than the content token budget")
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        if end < len(offsets):
            # Prefer a sentence boundary in the last quarter of this window.
            for candidate in range(end, max(start + overlap + 1, start + budget * 3 // 4), -1):
                if re.search(
                    r"[.!?]$", text[offsets[candidate - 1][0] : offsets[candidate - 1][1]]
                ):
                    end = candidate
                    break
        char_start, char_end = offsets[start][0], offsets[end - 1][1]
        chunk = text[char_start:char_end]
        # Re-tokenizing a substring can differ at its boundary. Shrink until it
        # actually fits rather than allowing the model to silently truncate.
        while len(tokenizer.encode(chunk, add_special_tokens=True)) > max_tokens:
            end -= 1
            if end <= start + overlap:
                raise ValueError("Tokenizer cannot produce a progressing chunk within budget")
            char_end = offsets[end - 1][1]
            chunk = text[char_start:char_end]
        yield chunk, char_start, char_end
        if end == len(offsets):
            break
        start = end - overlap


def build(cfg, analytical, embedder, limit=None):
    idx = cfg['index']
    if idx['max_tokens'] > embedder.max_seq_length:
        raise ValueError("max_tokens exceeds the embedding model sequence limit")
    budget = idx['max_tokens'] - embedder.tokenizer.num_special_tokens_to_add(pair=False)
    if idx['overlap_tokens'] >= budget:
        raise ValueError("overlap_tokens exceeds the model's content token budget")
    root = idx['dir']
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    run = root / 'builds' / run_id
    run.mkdir(parents=True)
    # Each build owns an independent store. Failed builds cannot affect readers.
    client = chromadb.PersistentClient(path=str(run / 'chroma'))
    collection = client.create_collection(
        'reviews', embedding_function=None, metadata={'hnsw:space': 'cosine'}
    )
    total = 0
    review_count = 0
    ids, docs, metas = [], [], []

    def flush():
        nonlocal total
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embedder.encode(docs))
        total += len(ids)
        print(f"  Indexed {total:,} chunks", flush=True)
        ids.clear()
        docs.clear()
        metas.clear()

    with duckdb.connect() as con:
        select_evidence(con, cfg, analytical, limit)
        # Trial catalog contains only restaurants represented in the trial.
        con.execute(
            "COPY (SELECT * FROM restaurants SEMI JOIN evidence USING(business_id)) TO "
            + sql_string(run / "restaurants.parquet")
            + " (FORMAT parquet, COMPRESSION zstd)"
        )
        cursor = con.execute("""SELECT e.review_id, e.user_id, e.business_id, e.text,
            e.date, e.stars, b.state, b.city, b.stars, b.price
            FROM evidence e JOIN restaurants b USING(business_id)""")
        while rows := cursor.fetchmany(idx['batch_size']):
            for review_id, user_id, bid, text, date, rating, state, city, stars, price in rows:
                review_count += 1
                for number, (chunk, begin, end) in enumerate(
                    chunks(text, embedder.tokenizer, idx['max_tokens'], idx['overlap_tokens'])
                ):
                    ids.append(f"{review_id}:{number}")
                    docs.append(chunk)
                    metadata = {
                        'review_id': review_id,
                        'user_id': user_id,
                        'business_id': bid,
                        'date': date.isoformat(),
                        'chunk_index': number,
                        'char_start': begin,
                        'char_end': end,
                    }
                    for key, value in {
                        'review_stars': rating,
                        'state': state,
                        'city': city,
                        'stars': stars,
                        'price': price,
                    }.items():
                        if value is not None:
                            metadata[key] = value
                    # Preserve existing price-filter semantics: unknown price is 0.
                    metadata.setdefault('price', 0)
                    metas.append(metadata)
                    if len(ids) == idx['batch_size']:
                        flush()
        if ids:
            flush()
    if not total or collection.count() != total:
        raise ValueError("Index is empty or stored chunk count does not match")
    manifest = {
        'version': 1,
        'run': run_id,
        'trial': limit is not None,
        'limit': limit,
        'analytical_run': analytical.name,
        'reviews': review_count,
        'chunks': total,
        'settings': {k: v for k, v in idx.items() if k != 'dir'},
        'serving': cfg['serving'],
        'model_max_seq_length': embedder.max_seq_length,
    }
    write_json(run / 'manifest.json', manifest)
    # Historical and trial builds cannot replace the live snapshot index.
    pointer = (
        'trial.json'
        if limit is not None
        else ('historical.json' if cfg['serving']['before'] else 'active.json')
    )
    write_json(root / pointer, manifest)
    return manifest


def load_index(cfg, pointer='active.json'):
    """Load a complete build only if model/chunk/filter settings still match."""
    root = cfg['index']['dir']
    manifest = json.loads((root / pointer).read_text(encoding='utf-8'))
    expected = {k: v for k, v in cfg['index'].items() if k != 'dir'}
    if (
        manifest['version'] != 1
        or manifest['settings'] != expected
        or manifest['serving'] != cfg['serving']
    ):
        raise ValueError("Index settings changed; rebuild before querying")
    run = root / 'builds' / manifest['run']
    collection = chromadb.PersistentClient(path=str(run / 'chroma')).get_collection(
        'reviews', embedding_function=None
    )
    if collection.count() != manifest['chunks']:
        raise ValueError("Stored index count does not match its completed build")
    return collection, run / 'restaurants.parquet', manifest


def search(collection, embedder, query, n_results=20, where=None):
    """Return chunks with their IDs, timestamps, and source offsets intact."""
    if n_results <= 0:
        raise ValueError("n_results must be positive")
    result = collection.query(
        query_embeddings=embedder.encode([query]),
        n_results=min(n_results, collection.count()),
        where=where,
        include=['documents', 'metadatas', 'distances'],
    )
    return [
        {**metadata, 'chunk_id': chunk_id, 'text': text, 'similarity': 1 - distance}
        for chunk_id, text, metadata, distance in zip(
            result['ids'][0],
            result['documents'][0],
            result['metadatas'][0],
            result['distances'][0],
            strict=True,
        )
    ]
