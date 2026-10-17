"""Filtered vector search over review snippets."""

from dataclasses import dataclass, field


@dataclass
class SearchFilters:
    state: str | None = None
    city: str | None = None
    max_price: int | None = None  # 1-4; restaurants with unknown price are kept
    min_stars: float | None = None
    business_ids: list[str] = field(default_factory=list)  # e.g. restaurants matching a cuisine


def build_where(filters: SearchFilters) -> dict | None:
    conditions = []
    if filters.state:
        conditions.append({"state": {"$eq": filters.state}})
    if filters.city:
        conditions.append({"city": {"$eq": filters.city}})
    if filters.max_price:
        conditions.append({"price": {"$lte": int(filters.max_price)}})
    if filters.min_stars:
        conditions.append({"stars": {"$gte": float(filters.min_stars)}})
    if filters.business_ids:
        conditions.append({"business_id": {"$in": list(filters.business_ids)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(collection, embedder, query: str, filters: SearchFilters, n_results: int) -> list[dict]:
    """Return the review snippets closest to the query, best first."""
    result = collection.query(
        query_embeddings=embedder.encode([query]),
        n_results=n_results,
        where=build_where(filters),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "business_id": meta["business_id"],
            "text": doc,
            "review_stars": meta.get("review_stars"),
            # Cosine distance -> similarity in [-1, 1], higher is closer.
            "similarity": 1.0 - dist,
        }
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
