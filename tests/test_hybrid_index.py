import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from recommender.index_hybrid import HybridIndex, build, source_chunks  # noqa: E402


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]}


class Embedder:
    tokenizer = Tokenizer()
    max_seq_length = 2048

    def encode(self, texts, query=False):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def inputs(tmp_path):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config_project.yaml').read_text())
    core = tmp_path / 'data/processed/analytical/fixture'
    prepared = tmp_path / 'data/processed/philadelphia/fixture'
    core.mkdir(parents=True)
    prepared.mkdir(parents=True)
    (core.parent / 'active.json').write_text(json.dumps({'run': 'fixture'}))
    catalog = pd.DataFrame(
        {
            'business_id': ['a', 'b', 'c'],
            'price': [1, None, 3],
            'delivery': [True, None, True],
            'takeout': [True, True, True],
            'outdoor_seating': [True, True, False],
        }
    )
    catalog.to_parquet(prepared / 'restaurants.parquet')
    reviews = pd.DataFrame(
        [
            ('r1', 'a', '2019-01-01', 5.0, 'quiet vegetarian curry'),
            ('r2', 'a', '2019-02-01', 1.0, 'loud crowded dinner'),
            ('r3', 'b', '2019-01-01', 4.0, 'vegetarian curry'),
            ('r4', 'c', '2021-01-01', 5.0, 'quiet vegetarian curry dinner'),
        ],
        columns=['review_id', 'business_id', 'date', 'stars', 'text'],
    )
    reviews['date'] = pd.to_datetime(reviews.date)
    reviews.to_parquet(core / 'reviews.parquet')
    membership = reviews[['review_id']].assign(
        evaluation_evidence=[True, True, True, False], serving_evidence=True
    )
    membership.to_parquet(prepared / 'index_membership.parquet')
    (prepared / 'readiness.json').write_text(
        json.dumps(
            {
                'decisions': cfg,
                'review_source': str((core / 'reviews.parquet').relative_to(tmp_path)),
            }
        )
    )
    return tmp_path, cfg


def test_build_reload_shared_filters_and_citations(inputs):
    root, cfg = inputs
    manifest = build(root, cfg, Embedder(), batch_size=2)
    assert manifest['reviews'] == 3  # Held-out review cannot enter evaluation index
    index = HybridIndex(root, cfg, Embedder(), mode='evaluation')
    results = index.search('vegetarian curry', max_price=2, delivery=True)
    assert [r['business_id'] for r in results] == ['a']
    assert {e['retriever'] for e in results[0]['evidence']} == {'bm25', 'gemma'}
    assert len({r['business_id'] for r in index.search('dinner')}) == 2
    assert all(e['retriever'] == 'gemma' for r in index.search('quasar') for e in r['evidence'])
    assert index.search('curry', max_price=1, outdoor=True)[0]['business_id'] == 'a'
    assert index.search('   ') == []
    for result in index.search('curry'):
        for evidence in result['evidence']:
            assert evidence['review_id'] != 'r4'
            assert evidence['text']
            assert evidence['source_end'] > evidence['source_start']
    cfg['retrieval']['fusion_k'] += 1
    with pytest.raises(ValueError, match='settings changed'):
        HybridIndex(root, cfg, Embedder(), mode='evaluation')


def test_trial_and_failed_build_cannot_replace_active(inputs):
    root, cfg = inputs
    original = build(root, cfg, Embedder())
    trial = build(root, cfg, Embedder(), limit=1)
    pointer = root / 'index/hybrid/evaluation/active.json'
    assert json.loads(pointer.read_text())['run'] == original['run']
    assert trial['trial'] and trial['reviews'] == 1

    class BrokenEmbedder(Embedder):
        def encode(self, texts, query=False):
            raise RuntimeError("Interrupted encoding")

    with pytest.raises(RuntimeError, match='Interrupted'):
        build(root, cfg, BrokenEmbedder())
    assert json.loads(pointer.read_text())['run'] == original['run']
    serving = build(root, cfg, Embedder(), mode='serving', batch_size=2)
    assert serving['reviews'] == 4


def test_prompt_budget_preserves_source_coverage(inputs):
    _, cfg = inputs
    cfg['evidence'].update(max_tokens=12, overlap_tokens=2)
    text = ' '.join(f'word{i}' for i in range(50))
    parts = source_chunks(text, Embedder(), cfg)
    assert len(parts) > 1
    assert parts[0][1] == 0 and parts[-1][2] == len(text)
    for part, start, end in parts:
        assert text[start:end] == part
        assert len(Tokenizer().encode(cfg['retrieval']['document_prompt'] + part)) <= 12
    for previous, following in zip(parts, parts[1:], strict=False):
        assert following[1] < previous[2]  # Overlap must not erase source spans
