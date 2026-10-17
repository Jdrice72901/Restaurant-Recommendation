"""Split reviews into snippets, embed them, and store them in a local Chroma index.

Each snippet keeps its business_id plus a few restaurant fields (state, city,
stars, price) as metadata, so searches can be filtered before ranking.
"""

from pathlib import Path
from typing import Protocol

import chromadb
import pandas as pd


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def chunk_text(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def open_collection(index_dir: Path, name: str, reset: bool = False):
    client = chromadb.PersistentClient(path=str(index_dir))
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    # Embeddings are always supplied by us, so Chroma's own embedder is never used.
    return client.get_or_create_collection(
        name=name, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )


def build_index(cfg: dict, embedder: Embedder, limit: int | None = None) -> int:
    processed: Path = cfg["data"]["processed_dir"]
    idx = cfg["index"]

    restaurants = pd.read_parquet(processed / "restaurants.parquet")
    reviews = pd.read_parquet(processed / "reviews.parquet")
    if limit:
        reviews = reviews.head(limit)

    meta_cols = restaurants.set_index("business_id")[["state", "city", "stars", "price"]]
    reviews = reviews.join(meta_cols, on="business_id", rsuffix="_restaurant")

    collection = open_collection(idx["dir"], idx["collection"], reset=True)

    ids, docs, metas = [], [], []
    for row in reviews.itertuples(index=False):
        for i, chunk in enumerate(chunk_text(row.text, idx["max_words_per_chunk"])):
            ids.append(f"{row.review_id}:{i}")
            docs.append(chunk)
            metas.append(
                {
                    "business_id": row.business_id,
                    "state": row.state,
                    "city": row.city,
                    "stars": float(row.stars_restaurant),
                    # Chroma metadata can't hold nulls; 0 means "price unknown".
                    "price": int(row.price) if pd.notna(row.price) else 0,
                    "review_stars": float(row.stars),
                }
            )

    batch = idx["batch_size"]
    for start in range(0, len(docs), batch):
        end = start + batch
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=embedder.encode(docs[start:end]),
        )
        print(f"  embedded {min(end, len(docs)):,} / {len(docs):,} snippets", end="\r", flush=True)
    print()
    return len(docs)
