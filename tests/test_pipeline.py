"""End-to-end check on a tiny fake Yelp dataset, so the full download isn't needed.

Uses a bag-of-words embedder instead of the real model so the test runs offline.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommender.answer import build_prompt, generate_answer  # noqa: E402
from recommender.config import load_config  # noqa: E402
from recommender.index import build_index, chunk_text, open_collection  # noqa: E402
from recommender.prep import build_tables  # noqa: E402
from recommender.rank import rank_restaurants  # noqa: E402
from recommender.retrieve import SearchFilters, build_where, search  # noqa: E402


class WordHashEmbedder:
    dims = 256

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dims
            for w in t.lower().split():
                w = w.strip(".,!?")
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


BUSINESSES = [
    # name, state, city, stars, review_count, is_open, categories, price
    ("b1", "Thai Orchid", "PA", "Philadelphia", 4.5, 120, 1, "Restaurants, Thai", "2"),
    ("b2", "Pizza Palace", "PA", "Philadelphia", 4.0, 300, 1, "Pizza, Restaurants", "u'1'"),
    ("b3", "Closed Thai", "PA", "Philadelphia", 5.0, 50, 0, "Restaurants, Thai", "2"),
    ("b4", "Hardware Store", "PA", "Philadelphia", 4.0, 80, 1, "Hardware, Shopping", None),
    ("b5", "Tampa Tacos", "FL", "Tampa", 4.5, 200, 1, "Mexican, Restaurants", "1"),
    ("b6", "Fancy Steak", "PA", "Pittsburgh", 3.5, 40, 1, "Steakhouses, Restaurants", "4"),
]

REVIEWS = {
    "b1": ["Amazing spicy pad thai and green curry, very authentic thai food", "Great thai curry, friendly staff"],
    "b2": ["Best pizza slice in the city, crispy crust", "Cheap pizza and fast service"],
    "b3": ["Spicy thai curry was perfect"],
    "b4": ["Bought a hammer"],
    "b5": ["Spicy tacos and fresh salsa"],
    "b6": ["Expensive steak, great wine list"],
}


def make_config(tmp_path):
    """Write the fake dataset under tmp_path and return a config pointing at it."""
    raw = tmp_path / "raw"
    raw.mkdir()
    with open(raw / "business.json", "w") as f:
        for bid, name, state, city, stars, rc, is_open, cats, price in BUSINESSES:
            attrs = {"RestaurantsPriceRange2": price} if price else None
            f.write(json.dumps({
                "business_id": bid, "name": name, "address": "1 Main St", "city": city,
                "state": state, "postal_code": "00000", "latitude": 40.0, "longitude": -75.0,
                "stars": stars, "review_count": rc, "is_open": is_open, "attributes": attrs,
                "categories": cats, "hours": None,
            }) + "\n")
    with open(raw / "review.json", "w") as f:
        n = 0
        for bid, texts in REVIEWS.items():
            for t in texts:
                n += 1
                f.write(json.dumps({
                    "review_id": f"r{n}", "user_id": "u1", "business_id": bid, "stars": 5.0,
                    "useful": n, "funny": 0, "cool": 0, "text": t, "date": "2020-01-01 00:00:00",
                }) + "\n")

    c = load_config()
    c["data"].update(raw_dir=raw, processed_dir=tmp_path / "processed",
                     business_file="business.json", review_file="review.json")
    c["filter"].update(states=["PA"], open_only=True, min_review_count=10)
    c["index"]["dir"] = tmp_path / "index"
    return c


@pytest.fixture()
def cfg(tmp_path):
    return make_config(tmp_path)


def test_prep_filters_to_open_restaurants_in_state(cfg):
    counts = build_tables(cfg)
    restaurants = pd.read_parquet(cfg["data"]["processed_dir"] / "restaurants.parquet")
    assert set(restaurants["business_id"]) == {"b1", "b2", "b6"}
    assert dict(zip(restaurants["business_id"], restaurants["price"])) == {"b1": 2, "b2": 1, "b6": 4}
    assert counts == {"restaurants": 3, "reviews": 5}


def test_search_and_rank_end_to_end(cfg):
    build_tables(cfg)
    embedder = WordHashEmbedder()
    assert build_index(cfg, embedder) == 5

    collection = open_collection(cfg["index"]["dir"], cfg["index"]["collection"])
    restaurants = pd.read_parquet(cfg["data"]["processed_dir"] / "restaurants.parquet").set_index("business_id")

    snippets = search(collection, embedder, "spicy thai curry", SearchFilters(state="PA"), 10)
    results = rank_restaurants(snippets, restaurants, cfg["ranking"], 3, 2)
    assert results[0]["name"] == "Thai Orchid"

    # Price filter drops the $$$$ steakhouse; city filter keeps only Philadelphia.
    snippets = search(collection, embedder, "steak", SearchFilters(state="PA", max_price=2), 10)
    assert "b6" not in {s["business_id"] for s in snippets}
    snippets = search(collection, embedder, "food", SearchFilters(state="PA", city="Pittsburgh"), 10)
    assert {s["business_id"] for s in snippets} == {"b6"}


def test_build_where_shapes():
    assert build_where(SearchFilters()) is None
    assert build_where(SearchFilters(state="PA")) == {"state": {"$eq": "PA"}}
    assert "$and" in build_where(SearchFilters(state="PA", min_stars=4))


def test_chunk_text_splits_long_reviews():
    assert chunk_text("a b c d e", 2) == ["a b", "c d", "e"]


def test_answer_prompt_and_no_llm():
    r = [{"name": "Thai Orchid", "city": "Philadelphia", "state": "PA", "stars": 4.5, "review_count": 120,
          "price": 2, "categories": "Thai", "snippets": ["great curry"]}]
    prompt = build_prompt("thai", r)
    assert "Thai Orchid" in prompt and "$$" in prompt and "great curry" in prompt
    assert list(generate_answer("thai", r, {"provider": "none"})) == []
    assert "No restaurants" in "".join(generate_answer("thai", [], {"provider": "claude"}))
