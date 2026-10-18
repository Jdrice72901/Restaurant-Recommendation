"""Streamlit front end. Run with:  streamlit run app.py"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommender.answer import MissingAPIKeyError, generate_answer  # noqa: E402
from recommender.config import load_config  # noqa: E402
from recommender.index import SentenceTransformerEmbedder, open_collection  # noqa: E402
from recommender.rank import rank_restaurants  # noqa: E402
from recommender.retrieve import SearchFilters, search  # noqa: E402

st.set_page_config(page_title="Restaurant Finder", page_icon="🍽️", layout="wide")

# Each person can keep their own key in .streamlit/secrets.toml (gitignored).
# An ANTHROPIC_API_KEY already set in the environment takes precedence.
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass


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
    # Every state in the index, largest first, labeled with its biggest city so the metro area is clear.
    by_state = restaurants.groupby("state")["city"].agg(["size", lambda c: c.mode().iat[0]])
    by_state.columns = ["count", "top_city"]
    by_state = by_state.sort_values("count", ascending=False)
    state = st.selectbox(
        "State",
        by_state.index.tolist(),
        format_func=lambda s: f"{s} · {by_state.at[s, 'top_city']} area ({by_state.at[s, 'count']:,} restaurants)",
    )
    # Cities in the chosen state, most restaurants first.
    city_counts = restaurants.loc[restaurants["state"] == state, "city"].value_counts()
    city = st.selectbox(
        "City",
        ["Any"] + city_counts.index.tolist(),
        format_func=lambda c: f"Any ({city_counts.sum():,} restaurants)" if c == "Any"
        else f"{c} ({city_counts[c]:,} restaurants)",
    )
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
        try:
            st.write_stream(generate_answer(query, results, cfg["llm"]))
        except MissingAPIKeyError as e:
            st.error(str(e))

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
