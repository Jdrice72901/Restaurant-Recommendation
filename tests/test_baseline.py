import importlib.util
import json
import re
from pathlib import Path

import duckdb
import pytest

from recommender.index_baseline import build, chunks, load_index, search
from recommender.prep_baseline import existing_tables, prepare, select_evidence

pipeline_spec = importlib.util.spec_from_file_location(
    'test_pipeline', Path(__file__).with_name('test_pipeline.py')
)
assert pipeline_spec is not None
assert pipeline_spec.loader is not None
pipeline = importlib.util.module_from_spec(pipeline_spec)
pipeline_spec.loader.exec_module(pipeline)
make_config = pipeline.make_config

spec = importlib.util.spec_from_file_location(
    'baseline_cli', Path(__file__).resolve().parents[1] / 'scripts/02_build_index.py'
)
assert spec is not None
assert spec.loader is not None
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]}

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))


class FakeEmbedder:
    tokenizer = Tokenizer()
    max_seq_length = 16

    def encode(self, texts):
        return [[1.0, 0.0, 0.0] for text in texts]


@pytest.fixture()
def cfg(tmp_path):
    old = make_config(tmp_path)
    return {
        'data': {**old['data'], 'analytical_dir': tmp_path / 'analytical'},
        'serving': {
            'states': [],
            'open_only': True,
            'min_review_count': 0,
            'reviews_per_restaurant': None,
            'before': None,
        },
        'index': {
            'dir': tmp_path / 'baseline',
            'embedding_model': 'fake',
            'revision': None,
            'max_tokens': 10,
            'overlap_tokens': 2,
            'batch_size': 2,
            'seed': 42,
        },
    }


def test_preparation_preserves_interactions_attributes_and_versions(cfg):
    run = prepare(cfg)
    with duckdb.connect() as con:
        restaurants = con.execute(
            "SELECT * FROM read_parquet(?)", [str(run / 'restaurants.parquet')]
        ).fetchdf()
        reviews = con.execute(
            "SELECT * FROM read_parquet(?)", [str(run / 'reviews.parquet')]
        ).fetchdf()
    assert set(restaurants.business_id) == {'b1', 'b2', 'b3', 'b5', 'b6'}
    assert {'attributes', 'hours', 'price'} <= set(restaurants.columns)
    assert len(reviews) == 7
    assert {'user_id', 'date', 'funny', 'cool'} <= set(reviews.columns)
    assert existing_tables(cfg) == run
    assert prepare(cfg) != run
    assert (run / 'reviews.parquet').exists()


def test_chunk_coverage_offsets_and_budget():
    text = 'One two three. Four five six seven eight nine ten eleven twelve.'
    pieces = list(chunks(text, Tokenizer(), 8, 2))
    covered = set()
    for value, start, end in pieces:
        assert value == text[start:end]
        assert len(Tokenizer().encode(value)) <= 8
        covered.update(range(start, end))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())


def test_empty_text_interactions_survive_and_paths_allow_apostrophes(cfg):
    source = cfg['data']['raw_dir'] / cfg['data']['review_file']
    row = json.loads(source.read_text().splitlines()[0])
    row.update(review_id='empty-text', text='   ')
    with source.open('a') as output:
        output.write(json.dumps(row) + '\n')
    cfg['data']['analytical_dir'] /= "user's data"
    run = prepare(cfg)
    with duckdb.connect() as con:
        assert (
            con.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(run / 'reviews.parquet')]
            ).fetchone()[0]
            == 8
        )
        cfg['serving']['reviews_per_restaurant'] = 1
        select_evidence(con, cfg, run)
        assert con.execute("SELECT count(*) FROM evidence").fetchone()[0] == 4
        assert con.execute("SELECT count(*) FROM evidence WHERE trim(text) = ''").fetchone()[0] == 0


def test_trial_failure_and_settings_do_not_invalidate_active(cfg):
    run = prepare(cfg)
    embedder = FakeEmbedder()
    result = build(cfg, run, embedder)
    active = (cfg['index']['dir'] / 'active.json').read_bytes()
    collection, catalog, manifest = load_index(cfg)
    assert catalog.exists() and manifest == result
    evidence = search(collection, embedder, 'curry', 2)
    assert {'review_id', 'date', 'char_start', 'chunk_id', 'user_id'} <= evidence[0].keys()
    build(cfg, run, embedder, limit=2)
    assert (cfg['index']['dir'] / 'active.json').read_bytes() == active
    first_trial = json.loads((cfg['index']['dir'] / 'trial.json').read_text())
    assert first_trial['reviews'] == 2

    class Broken(FakeEmbedder):
        def encode(self, texts):
            raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match='embedding failed'):
        build(cfg, run, Broken())
    assert (cfg['index']['dir'] / 'active.json').read_bytes() == active
    cfg['index']['max_tokens'] += 1
    with pytest.raises(ValueError, match='settings changed'):
        load_index(cfg)


def test_historical_evidence_uses_cutoff_and_train_aggregates(cfg):
    run = prepare(cfg)
    cfg['serving'].update(before='2020-01-02', open_only=False)
    with duckdb.connect() as con:
        select_evidence(con, cfg, run)
        assert con.execute("SELECT count(*) FROM evidence").fetchone()[0] == 7
        assert con.execute("SELECT max(review_count) FROM restaurants").fetchone()[0] == 2
        assert con.execute("SELECT min(stars) FROM restaurants").fetchone()[0] == 5
    cfg['serving']['before'] = '2020-01-01'
    with duckdb.connect() as con:
        select_evidence(con, cfg, run)
        assert con.execute("SELECT count(*) FROM evidence").fetchone()[0] == 0


def test_config_rejects_invalid_numeric_values(tmp_path):
    import yaml

    cfg = yaml.safe_load((cli.REPO_ROOT / 'config_index.yaml').read_text())
    cfg['index']['batch_size'] = 0
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='positive integer'):
        cli.load_settings(path)
