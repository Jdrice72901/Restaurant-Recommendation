# Restaurant-Recommendation

LLM powered restaurant recommendation system, completed as a group project.

Link to the data set used
https://business.yelp.com/data/resources/open-dataset/
The 4.35 GB ZIP contains a documentation PDF and a gzip compressed TAR. The TAR contains another agreement PDF and five JSONL files (9.29 GB extracted in total). Takes a while to download and uncompress.

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

**Real data:** download, extract, and validate the source package:

```bash
python scripts\00_data_ingestion.py
```

The script resolves paths from its repository location, so it also works when invoked by absolute path from another working directory. It downloads to `data/raw/Yelp-JSON.zip` on the first run and reuses that archive on later runs. Files elsewhere, including the repository root, are ignored unless explicitly selected with `--archive`.

The nested TAR is streamed directly into `data/raw/` without saving an intermediate archive. Both PDFs are preserved. Validation checks expected payloads, extracted sizes, PDF headers, and every JSON record's syntax and required fields. Existing raw files are retained and validated; size conflicts cause an error. Interrupted writes use `.part` files. Preparation into Parquet and embedding remain separate steps.

If the direct download is blocked, download the package manually from Yelp and use:

```bash
python scripts\00_data_ingestion.py --archive data/raw/Yelp-JSON.zip
```

`--archive` and `--config` paths are relative to the repository root unless absolute. Full validation reads the entire extracted dataset and can take several minutes. The existing recommendation pipeline currently consumes only these two source files:

```text
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

### Philadelphia hybrid workflow

After the audited EDA exports exist, finalize the city inputs and build BM25 + Gemma:

```powershell
python scripts/01_data_readiness.py
python scripts/02_build_index.py --mode evaluation --limit 64 --query "vegetarian dinner"
python scripts/02_build_index.py --mode evaluation  # full training-evidence index
python scripts/02_build_index.py --mode serving     # full historical serving index
```

The trial is a build/reload smoke test. Both full builds encode all eligible reviews and can take substantial time on CPU. `--batch-size` controls the number of passages written per batch; `--device cpu` or `--device cuda` selects the embedding device. Gemma uses the existing Hugging Face access/accepted model terms. `--config` defaults to `config_project.yaml`.

`--embedding-batch-size` independently controls model inference batches (default 32).

If you have a modern NVidia GPU, the CUDA workflow is many times faster.

Artifacts live under `index/hybrid/<evaluation|serving>/builds/<build-id>/`: Chroma, BM25, source passages, restaurant metadata and a completed manifest. `trial.json` and `active.json` are separate pointers; evaluation and serving cannot overwrite each other. Failed builds do not replace a completed pointer. A failed build currently restarts in a new build directory; automatic resume is not implemented.

`recommender.index_hybrid.HybridIndex` reloads these artifacts and returns fused unique restaurants with contributing source passages. Price/delivery/takeout/outdoor filters apply to both retrievers, with unknown attributes excluded from strict requirements. The existing Streamlit app still is unchanged in function and index usage.

### Earlier dense only baseline workflow

With `--baseline`, `scripts/02_build_index.py` uses `config_index.yaml`, `prep_baseline.py`, and `index_baseline.py`. The original workflow below remains available. The existing Streamlit app still uses the original configuration/index; it has not been switched to the new baseline.

```powershell
python scripts/02_build_index.py --baseline --prep-only
python scripts/02_build_index.py --baseline --skip-prep --limit 2000  # isolated trial
python scripts/02_build_index.py --baseline --skip-prep               # full serving build
```

Paths, including `--config`, resolve against the repository root from any working directory. Preparation can run without loading/downloading the embedding model. Trial limits apply only to evidence indexing, not the full analytical preparation. A full uncapped build can embed millions of reviews; use a trial first.

Preparation retains all businesses with the exact `Restaurants` category token, regardless of state, open status, or review count. It preserves all business fields (including attributes/hours), adds price, and retains all their review interactions, including user IDs, full text, ratings, votes, and timestamps. Empty review text remains in the analytical data but is excluded from embedding. Original raw files are never modified. Other business categories and user/tip/checkin files remain available in raw storage; these tables are a restaurant-domain analytical core.

```text
data/processed/analytical/
  active.json
  <version>/
    restaurants.parquet
    reviews.parquet
    manifest.json
index/baseline/
  active.json             # only a successful full snapshot build updates this
  trial.json              # limited experiments
  historical.json         # full cutoff-based evidence builds
  builds/<version>/
    chroma/
    restaurants.parquet
    manifest.json
```

Every preparation/index build writes to a new version; prior versions and failed builds remain available for inspection. No automatic deletion is performed. The small active pointer file is replaced only after successful completion. The index manifest ties each build to its analytical version, configuration, and counts. `--skip-prep` reuses the last completed analytical version after checking its source paths/schema version. No file hashes are used; raw inputs are treated as immutable.

Serving configuration controls state, snapshot open status, minimum review count, and an optional per-restaurant evidence cap. These never change the analytical core. The evidence cap defaults to null (all eligible text); when set, selection is most recent first with review ID breaking ties. Token aware chunks preserve source offsets, prefer sentence boundaries, include overlap, and are checked against the model's context length. Embedding and database writes use bounded batches.

For a historical experiment, set `serving.before` to an exclusive ISO timestamp and `open_only: false`. Only earlier reviews become evidence, and restaurant ratings and counts are recomputed from earlier interactions; useful votes are not used for selection. **This is not a finished evaluation split:** restaurant identity, attributes, geography, and category membership still come from the dataset snapshot. A defensible temporal benchmark and training-only feature construction remain separate work. Historical builds cannot replace the snapshot serving pointer.

`recommender.index_baseline.load_index(cfg)` returns the completed collection, its matching restaurant catalog path, and its manifest, rejecting mismatched model/chunk/filter settings. `search()` preserves review IDs, dates, user IDs, chunk IDs, offsets, and similarity. A future app integration must use these helpers and carry source attribution through ranking and answer generation. The original app does not automatically gain these capabilities. Pin `index.revision` to a model commit when freezing experiments; null uses the model repository's default revision.

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
| --- | --- | --- |
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

## Troubleshooting

- **`No module named ...` or numpy errors when using conda:** your terminal may be using conda's Python
  instead of the project's. Running `.venv\Scripts\python.exe` directly (as in step 6) always uses the
  right one.
- **`pip install` times out:** rerun the same command. Packages that already downloaded are reused.
- **"No index found" in the app:** run step 4 first.
- **"No Claude API key provided":** add your key (step 5), or set `provider` to `none` or `ollama`.
- **Changed `config.yaml` or `secrets.toml` and nothing happened:** the app reads them at startup, so stop
  it (Ctrl+C) and run it again.
