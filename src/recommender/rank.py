"""Group matching snippets by restaurant and score each restaurant."""

import math

import pandas as pd


def rank_restaurants(
    snippets: list[dict],
    restaurants: pd.DataFrame,
    weights: dict,
    top_n: int,
    snippets_per_restaurant: int,
) -> list[dict]:
    """Score = similarity of the best few snippets, plus star rating, plus review volume.

    `restaurants` is the restaurants table indexed by business_id.
    """
    by_business: dict[str, list[dict]] = {}
    for s in snippets:
        by_business.setdefault(s["business_id"], []).append(s)

    candidates = [b for b in by_business if b in restaurants.index]
    if not candidates:
        return []
    max_reviews = max(restaurants.loc[candidates, "review_count"].max(), 1)

    ranked = []
    for business_id in candidates:
        matches = sorted(by_business[business_id], key=lambda s: s["similarity"], reverse=True)
        top = matches[:snippets_per_restaurant]
        info = restaurants.loc[business_id]

        similarity = sum(s["similarity"] for s in top) / len(top)
        stars = (info["stars"] - 1) / 4
        popularity = math.log1p(info["review_count"]) / math.log1p(max_reviews)
        score = (
            weights["weight_similarity"] * similarity
            + weights["weight_stars"] * stars
            + weights["weight_popularity"] * popularity
        )

        ranked.append(
            {
                "business_id": business_id,
                "name": info["name"],
                "address": info["address"],
                "city": info["city"],
                "state": info["state"],
                "stars": float(info["stars"]),
                "review_count": int(info["review_count"]),
                "price": None if pd.isna(info["price"]) else int(info["price"]),
                "categories": info["categories"],
                "latitude": float(info["latitude"]),
                "longitude": float(info["longitude"]),
                "score": score,
                "similarity": similarity,
                "snippets": [s["text"] for s in top],
            }
        )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]
