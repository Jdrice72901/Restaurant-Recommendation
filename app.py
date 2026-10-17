"""Streamlit front end. Run with:  streamlit run app.py"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommender.answer import generate_answer  # noqa: E402
from recommender.config import load_config  # noqa: E402
from recommender.index import SentenceTransformerEmbedder, open_collection  # noqa: E402
from recommender.rank import rank_restaurants  # noqa: E402
from recommender.retrieve import SearchFilters, search  # noqa: E402

st.set_page_config(page_title="Restaurant Finder", page_icon="🍽️", layout="wide")


@st.cache_resource
def load_resources():
    cfg = load_config()
    restaurants_path = cfg["data"]["processed_dir"] / "restaurants.parquet"
    if not restaurants_path.exists():
        return cfg, None, None, None
    restaurants = pd.read_parquet(restaurants_path).set_index("business_id")
    embedder = SentenceTransformerEmbedder(cfg["index"]["embedding_model"])
    collection = open_collection(cfg["index"]["dir"], cfg["index"]["collection"])
    return cfg, restaurants, embedder, collection


cfg, restaurants, embedder, collection = load_resources()

st.title("🍽️ Restaurant Finder")
st.caption("Describe what you're in the mood for. Results come from Yelp reviews.")

if restaurants is None:
    st.error("No index found. Run `python scripts/build_index.py` first (see README).")
    st.stop()

with st.sidebar:
    st.header("Filters")
    states = sorted(restaurants["state"].dropna().unique())
    state = st.selectbox("State", states)
    cities = sorted(restaurants.loc[restaurants["state"] == state, "city"].dropna().unique())
    city = st.selectbox("City", ["Any"] + cities)
    cuisine = st.text_input("Cuisine (optional)", placeholder="e.g. Thai, Pizza, Vegan")
    max_price = st.select_slider("Max price", options=[1, 2, 3, 4], value=4, format_func=lambda p: "$" * p)
    min_stars = st.slider("Minimum stars", 1.0, 5.0, 3.5, 0.5)
    top_n = st.slider("Number of results", 3, 20, cfg["search"]["top_restaurants"])

query = st.text_input("What are you looking for?", placeholder="Cozy spot with great ramen and quick service")

if st.button("Find restaurants", type="primary") and query.strip():
    filters = SearchFilters(
        state=state,
        city=None if city == "Any" else city,
        max_price=max_price if max_price < 4 else None,
        min_stars=min_stars,
    )
    if cuisine.strip():
        matches = restaurants["categories"].str.contains(cuisine.strip(), case=False, na=False, regex=False)
        filters.business_ids = restaurants.index[matches & (restaurants["state"] == state)].tolist()
        if not filters.business_ids:
            st.warning(f"No restaurants in {state} are listed under '{cuisine}'.")
            st.stop()

    with st.spinner("Searching reviews..."):
        snippets = search(collection, embedder, query, filters, cfg["search"]["candidate_chunks"])
        results = rank_restaurants(
            snippets, restaurants, cfg["ranking"], top_n, cfg["search"]["snippets_per_restaurant"]
        )

    if cfg["llm"]["provider"] != "none":
        st.subheader("Recommendation")
        st.write_stream(generate_answer(query, results, cfg["llm"]))

    if results:
        st.subheader(f"Top {len(results)} matches")
        st.map(pd.DataFrame(results)[["latitude", "longitude"]], size=40)
        for r in results:
            with st.container(border=True):
                price = "$" * r["price"] if r["price"] else "price n/a"
                st.markdown(f"### {r['name']}")
                st.markdown(
                    f"⭐ **{r['stars']}** · {r['review_count']:,} reviews · {price} · "
                    f"{r['address']}, {r['city']}, {r['state']}"
                )
                st.caption(r["categories"])
                for s in r["snippets"]:
                    st.markdown(f"> {s[:400]}{'…' if len(s) > 400 else ''}")
    else:
        st.info("No restaurants matched. Try loosening the filters.")
