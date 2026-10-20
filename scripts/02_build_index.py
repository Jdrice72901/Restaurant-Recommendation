import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from recommender.prep_baseline import existing_tables, prepare  # noqa: E402


def load_settings(path=None):
    path = Path(path) if path else Path('config_index.yaml')

    if not path.is_absolute():
        path = REPO_ROOT / path

    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    for section, key in (('data', 'raw_dir'), ('data', 'analytical_dir'), ('index', 'dir')):
        value = Path(cfg[section][key])
        cfg[section][key] = value if value.is_absolute() else REPO_ROOT / value

    idx, serving = cfg['index'], cfg['serving']
    for key in ('max_tokens', 'batch_size'):
        if type(idx[key]) is not int or idx[key] <= 0:
            raise ValueError(f"index.{key} must be a positive integer")
        
    if type(idx['overlap_tokens']) is not int or not 0 <= idx['overlap_tokens'] < idx['max_tokens']:
        raise ValueError("overlap_tokens must be nonnegative and smaller than max_tokens")
    
    if type(idx['seed']) is not int:
        raise ValueError("index.seed must be an integer")
    
    if type(serving['min_review_count']) is not int or serving['min_review_count'] < 0:
        raise ValueError("min_review_count must be nonnegative")
    
    cap = serving['reviews_per_restaurant']
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError("reviews_per_restaurant must be null or a positive integer")
    
    if not isinstance(serving['states'], list) or not all(
        isinstance(s, str) for s in serving['states']
    ):
        raise ValueError("states must be a list of strings")
    
    if type(serving['open_only']) is not bool:
        raise ValueError("open_only must be true or false")
    
    if serving['before'] is not None:
        value = datetime.fromisoformat(str(serving['before']))
        if value.tzinfo is not None:
            raise ValueError("before must use naive Yelp timestamps (no timezone)")
        serving['before'] = value.isoformat()
        if serving['open_only']:
            raise ValueError("Historical builds require open_only: false")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='Config path relative to repository root')
    parser.add_argument('--baseline', action='store_true', help='Use the previous dense-only baseline')
    parser.add_argument('--mode', choices=['evaluation', 'serving'], default='evaluation')
    parser.add_argument('--batch-size', type=int, default=64, help='Passages per index-write batch')
    parser.add_argument(
        '--embedding-batch-size',
        type=int,
        default=32,
        help='Passages per model inference batch; independent of index writes',
    )
    parser.add_argument('--device', help='Sentence Transformers device, e.g. cpu or cuda')
    parser.add_argument('--query', help='Run a build/reload smoke query after building')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--skip-prep', action='store_true', help='Reuse complete analytical tables')
    mode.add_argument(
        '--prep-only',
        action='store_true',
        help='Prepare analytical tables without loading the model',
    )
    parser.add_argument('--limit', type=int, help='Deterministic trial review sample; never replaces active index')

    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')

    if args.prep_only and args.limit is not None:
        parser.error('--limit applies to indexing, not preparation')

    if not args.baseline:
        if args.prep_only or args.skip_prep:
            parser.error('Use --baseline for legacy preparation flags; hybrid builds consume 01_data_readiness.py outputs')

        if args.batch_size <= 0 or args.embedding_batch_size <= 0:
            parser.error('--batch-size and --embedding-batch-size must be positive')

        from recommender.index_hybrid import GemmaEmbedder, HybridIndex, build
        from recommender.index_hybrid import load_settings as hybrid_settings

        cfg = hybrid_settings(REPO_ROOT, args.config)
        embedder = GemmaEmbedder(cfg, device=args.device, batch_size=args.embedding_batch_size)
        manifest = build(REPO_ROOT, cfg, embedder, args.mode, args.limit, args.batch_size)
        print(f"Completed {args.mode} {'trial' if args.limit else 'full'} hybrid build: {manifest['passages']:,} passages")

        if args.query:
            import json

            index = HybridIndex(REPO_ROOT, cfg, embedder, args.mode, trial=args.limit is not None)
            print(json.dumps(index.search(args.query)[:3], indent=2))
        return
    
    cfg = load_settings(args.config)
    analytical = existing_tables(cfg) if args.skip_prep else prepare(cfg)
    print(f"Analytical tables: {analytical}", flush=True)

    if args.prep_only:
        return
    
    from recommender.index_baseline import Embedder, build

    if cfg['serving']['before']:
        print(
            "Historical evidence mode: business attributes/geography remain snapshot data; this is not a complete evaluation split."
        )
    manifest = build(cfg, analytical, Embedder(cfg['index']), args.limit)
    print(f"Completed {'trial' if manifest['trial'] else 'full'} build: {manifest['chunks']:,} chunks")


if __name__ == '__main__':
    main()

"""
Examples:
    python scripts/02_build_index.py --mode evaluation --limit 128
    python scripts/02_build_index.py --mode serving
    python scripts/02_build_index.py --baseline --prep-only
"""