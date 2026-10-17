"""Build everything the app needs, in one command.

    python scripts/build_index.py              # prep tables + build the search index
    python scripts/build_index.py --skip-prep  # rebuild only the index
    python scripts/build_index.py --limit 2000 # quick trial on the first 2,000 reviews
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommender.config import load_config  # noqa: E402
from recommender.index import SentenceTransformerEmbedder, build_index  # noqa: E402
from recommender.prep import build_tables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to config.yaml (default: repo root)")
    parser.add_argument("--skip-prep", action="store_true", help="reuse existing Parquet tables")
    parser.add_argument("--limit", type=int, help="only index the first N reviews")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not args.skip_prep:
        t = time.time()
        print("Filtering restaurants and their reviews...")
        counts = build_tables(cfg)
        print(f"  {counts['restaurants']:,} restaurants, {counts['reviews']:,} reviews ({time.time() - t:.0f}s)")

    t = time.time()
    print(f"Embedding reviews with {cfg['index']['embedding_model']}...")
    embedder = SentenceTransformerEmbedder(cfg["index"]["embedding_model"])
    n = build_index(cfg, embedder, limit=args.limit)
    print(f"  indexed {n:,} snippets into {cfg['index']['dir']} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
