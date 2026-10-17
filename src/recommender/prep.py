"""Turn the raw Yelp JSON files into two clean Parquet tables.

restaurants.parquet: one row per restaurant (business grain)
reviews.parquet:     the reviews kept for those restaurants, joined on business_id

DuckDB reads the newline-delimited JSON directly, so the multi-GB review file
never has to fit in memory.
"""

from pathlib import Path

import duckdb

BUSINESS_COLUMNS = {
    "business_id": "VARCHAR",
    "name": "VARCHAR",
    "address": "VARCHAR",
    "city": "VARCHAR",
    "state": "VARCHAR",
    "postal_code": "VARCHAR",
    "latitude": "DOUBLE",
    "longitude": "DOUBLE",
    "stars": "DOUBLE",
    "review_count": "INTEGER",
    "is_open": "INTEGER",
    "attributes": "JSON",
    "categories": "VARCHAR",
}

REVIEW_COLUMNS = {
    "review_id": "VARCHAR",
    "user_id": "VARCHAR",
    "business_id": "VARCHAR",
    "stars": "DOUBLE",
    "useful": "INTEGER",
    "funny": "INTEGER",
    "cool": "INTEGER",
    "text": "VARCHAR",
    "date": "VARCHAR",
}


def _read_json(path: Path, columns: dict) -> str:
    cols = ", ".join(f"'{k}': '{v}'" for k, v in columns.items())
    return f"read_json('{path.as_posix()}', format='newline_delimited', columns={{{cols}}})"


def build_tables(cfg: dict) -> dict:
    data = cfg["data"]
    filt = cfg["filter"]
    raw_dir: Path = data["raw_dir"]
    out_dir: Path = data["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    business_path = raw_dir / data["business_file"]
    review_path = raw_dir / data["review_file"]
    for p in (business_path, review_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Put the Yelp JSON files in {raw_dir}.")

    con = duckdb.connect()

    conditions = ["categories ILIKE '%Restaurants%'", f"review_count >= {int(filt['min_review_count'])}"]
    if filt.get("open_only"):
        conditions.append("is_open = 1")
    states = filt.get("states") or []
    if states:
        state_list = ", ".join(f"'{s}'" for s in states)
        conditions.append(f"state IN ({state_list})")

    # Price is stored as a stringified attribute ("2", "u'2'", "None"), so pull out the digit.
    con.execute(f"""
        CREATE TABLE restaurants AS
        SELECT
            business_id, name, address, city, state, postal_code,
            latitude, longitude, stars, review_count, is_open, categories,
            TRY_CAST(regexp_extract(json_extract_string(attributes, '$.RestaurantsPriceRange2'), '[1-4]') AS INTEGER)
                AS price
        FROM {_read_json(business_path, BUSINESS_COLUMNS)}
        WHERE {' AND '.join(conditions)}
    """)

    # Keep the most useful (then most recent) reviews per restaurant.
    con.execute(f"""
        CREATE TABLE reviews AS
        SELECT r.review_id, r.business_id, r.stars, r.useful, r.text, r.date
        FROM {_read_json(review_path, REVIEW_COLUMNS)} r
        SEMI JOIN restaurants b ON r.business_id = b.business_id
        WHERE r.text IS NOT NULL AND length(trim(r.text)) > 0
        QUALIFY row_number() OVER (
            PARTITION BY r.business_id ORDER BY r.useful DESC, r.date DESC
        ) <= {int(filt['reviews_per_restaurant'])}
    """)

    restaurants_out = out_dir / "restaurants.parquet"
    reviews_out = out_dir / "reviews.parquet"
    con.execute(f"COPY restaurants TO '{restaurants_out.as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY reviews TO '{reviews_out.as_posix()}' (FORMAT parquet)")

    counts = {
        "restaurants": con.execute("SELECT count(*) FROM restaurants").fetchone()[0],
        "reviews": con.execute("SELECT count(*) FROM reviews").fetchone()[0],
    }
    con.close()
    return counts
