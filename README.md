# Restaurant-Recommendation
LLM powered restaurant recommendation system, completed as a group project.

Link to the data set used
https://business.yelp.com/data/resources/open-dataset/
This is a large file, compressed TAR file (4.35 GB). Uncompressed it contains 1 PDF and 5 JSON files (8.65 GB)
Takes a while to download and uncompress.

## Getting started

Commands below are for Windows PowerShell. On Mac/Linux, activate with `source .venv/bin/activate` and use
forward slashes in paths.

### 1. Get the code

Accept the GitHub collaborator invite, then clone the repo (or use **Clone Repository** in VS Code):

```bash
git clone https://github.com/Jdrice72901/Restaurant-Recommendation.git
cd Restaurant-Recommendation
git checkout first-version
```

(Skip the `checkout` once this branch is merged into `main`.)

### 2. Install the dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --default-timeout=120 -r requirements.txt
```

This downloads about 1 GB (torch is the big one). The longer timeout helps on slow connections.

### 3. Add the data

**Real data:** copy these two files from the Yelp download into `data\raw\`:

```
data\raw\yelp_academic_dataset_business.json
data\raw\yelp_academic_dataset_review.json
```

**No download yet?** Generate a small made-up dataset in the same format (about 50 businesses and 900
reviews) so you can try the whole pipeline:

```bash
python scripts\make_sample_data.py
```

When the real files arrive, copy them over the sample ones and rebuild (step 4).

The `data\` folder is gitignored, so the dataset stays on your machine and is never pushed.

### 4. Build the search index

```bash
python scripts\build_index.py --limit 2000   # quick trial on the first 2,000 reviews
python scripts\build_index.py                # full build for the states in config.yaml
```

Run it once, and again whenever the data or the `filter` settings in `config.yaml` change. The first run also
downloads the embedding model (about 90 MB).

### 5. Choose how recommendations are written

Set `llm.provider` in `config.yaml`:

| provider | What you get | Needs |
|---|---|---|
| `none` | Ranked restaurants, map and review snippets. Fastest. | Nothing |
| `claude` (default) | The above, plus a written recommendation from Claude Haiku 4.5. A few seconds per search. | An Anthropic API key |
| `ollama` | The above, written by a model running on your own computer. Slower; depends on your hardware. | Ollama and a downloaded model |

**Claude:** get a key at https://console.anthropic.com (API Keys). Copy `.streamlit\secrets.toml.example` to
`.streamlit\secrets.toml` and paste your key in:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

`secrets.toml` is gitignored, so each person keeps their own key and it never gets pushed. (Setting
`ANTHROPIC_API_KEY` as an environment variable also works.)

**Ollama:** install it from https://ollama.com, download a model (for example `ollama pull llama3.1`), and run
`ollama list` to see the models you have. Then put that name in `config.yaml`:

```yaml
llm:
  provider: ollama
  ollama_model: llama3.1    # or whatever Ollama model you have installed
```

### 6. Run the app

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

It opens at http://localhost:8501. Pick a state in the sidebar and describe what you're in the mood for.
The first page load takes about 15 seconds while the embedding model loads.

## Troubleshooting

- **`No module named ...` or numpy errors when using conda:** your terminal may be using conda's Python
  instead of the project's. Running `.venv\Scripts\python.exe` directly (as in step 6) always uses the
  right one.
- **`pip install` times out:** rerun the same command. Packages that already downloaded are reused.
- **"No index found" in the app:** run step 4 first.
- **"No Claude API key provided":** add your key (step 5), or set `provider` to `none` or `ollama`.
- **Changed `config.yaml` or `secrets.toml` and nothing happened:** the app reads them at startup, so stop
  it (Ctrl+C) and run it again.

## How it works

This is a retrieval-augmented generation (RAG) pipeline:

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
5. **Answer** (`src/recommender/answer.py`): sends only the top restaurants and their snippets to the LLM
   chosen in step 5, which writes the recommendation.
6. **App** (`app.py`): the Streamlit page.

Steps 1 and 2 run in `scripts/build_index.py`; steps 3 to 5 run on every search.

## Configuration

Everything tunable lives in `config.yaml`:

- `filter`: which states to index (`states: [PA]`; `[]` means all), open restaurants only, minimum review
  count, and reviews kept per restaurant. Rebuild the index after changing these.
- `index`: embedding model and snippet length. Rebuild after changing these too.
- `search`: how many snippets to pull per search, restaurants to show, and snippets per restaurant.
- `ranking`: weights for match strength, star rating and review volume.
- `llm`: provider and model names (see step 5).

## Tests

```bash
.venv\Scripts\python.exe -m pytest
```

The tests build a tiny fake dataset, so they run without the Yelp download.
