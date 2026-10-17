# Restaurant-Recommendation
An LLM powered restaurant recommendation system, completed as a group project.

Link to the data set used
https://business.yelp.com/data/resources/open-dataset/
This is a large file, compressed TAR file (4.35 GB). Uncompressed it contains 1 PDF and 5 JSON files (8.65 GB)
Takes a while to download and uncompress.

## How it works

1. **Prep** (`src/recommender/prep.py`): keeps open restaurants from `business.json` for the states in
   `config.yaml`, then keeps the top reviews for each one from `review.json`, joined on `business_id`.
   Output: `data/processed/restaurants.parquet` and `reviews.parquet`.
2. **Index** (`src/recommender/index.py`): splits reviews into snippets, embeds them with
   sentence-transformers (`all-MiniLM-L6-v2`), and saves them in a local Chroma index under `index/`.
   Each snippet carries its restaurant's `business_id`, state, city, stars and price.
3. **Search** (`src/recommender/retrieve.py`): filters by state, city, price, stars and cuisine, then finds
   the review snippets closest in meaning to the user's request.
4. **Rank** (`src/recommender/rank.py`): groups snippets by restaurant and scores each one on match
   strength, star rating and review volume (weights in `config.yaml`).
5. **Answer** (`src/recommender/answer.py`): sends only the top restaurants and their snippets to an LLM,
   which writes the recommendation. Claude Haiku 4.5 by default; set `llm.provider` to `ollama` to use a
   local model, or `none` to skip it.
6. **App** (`app.py`): the Streamlit page.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the Yelp files in `data/raw/`:

```
data/raw/yelp_academic_dataset_business.json
data/raw/yelp_academic_dataset_review.json
```

Build the tables and the search index (one time; rerun after changing `filter` settings):

```bash
python scripts/build_index.py --limit 2000   # quick trial run
python scripts/build_index.py                # full build for the states in config.yaml
```

Set your Claude API key and start the app:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

## Using Ollama instead of Claude

Install a model (`ollama pull llama3.1`), then in `config.yaml` set:

```yaml
llm:
  provider: ollama
  ollama_model: llama3.1
```

## Tests

```bash
pytest
```

The tests build a tiny fake dataset, so they run without the Yelp download.
