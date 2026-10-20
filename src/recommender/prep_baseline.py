import json
from uuid import uuid4

import duckdb


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def prepare(cfg):
    """Create a new immutable analytical version; promote only complete tables."""
    data = cfg['data']
    root = data['analytical_dir']
    run = root / uuid4().hex
    run.mkdir(parents=True)
    business = data['raw_dir'] / data['business_file']
    reviews = data['raw_dir'] / data['review_file']
    
    for source in (business, reviews):
        if not source.is_file():
            raise FileNotFoundError(source)
        
    with duckdb.connect() as con:
        # Business JSON is small, preserve all fields. Category matching uses comma separated token
        con.execute(f"""
            CREATE TABLE restaurants AS
            SELECT *, TRY_CAST(json_extract_string(to_json(attributes),
                '$.RestaurantsPriceRange2') AS INTEGER) AS price
            FROM read_json_auto({sql_string(business)}, format='newline_delimited')
            WHERE list_contains(list_transform(string_split(categories, ','),
                x -> lower(trim(x))), 'restaurants')
        """)

        con.execute(f"""
            CREATE VIEW reviews AS
            SELECT r.* REPLACE (CAST(r.date AS TIMESTAMP) AS date)
            FROM read_json({sql_string(reviews)}, format='newline_delimited', columns={{
                review_id:'VARCHAR', user_id:'VARCHAR', business_id:'VARCHAR',
                stars:'DOUBLE', useful:'INTEGER', funny:'INTEGER', cool:'INTEGER',
                text:'VARCHAR', date:'VARCHAR'}}) r
            SEMI JOIN restaurants b USING (business_id)
        """)
        
        counts = {}
        for table, key in (('restaurants', 'business_id'), ('reviews', 'review_id')):
            path = run / f'{table}.parquet'
            con.execute(f"COPY {table} TO {sql_string(path)} (FORMAT parquet, COMPRESSION zstd)")
            result = con.execute(
                f"SELECT count(*), count(DISTINCT {key}), count({key}) FROM read_parquet(?)",
                [str(path)],
            ).fetchone()

            if result is None:
                raise ValueError(f"{table}: unable to validate table")
            
            count, distinct, nonnull = result
            if not count or count != distinct or count != nonnull:
                raise ValueError(f"{table}: empty table or invalid/duplicate identifiers")
            
            counts[table] = count

        invalid_reviews = con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE user_id IS NULL OR date IS NULL",
            [str(run / 'reviews.parquet')],
        ).fetchone()

        if invalid_reviews is None:
            raise ValueError("Reviews: unable to validate required fields")
        
        if invalid_reviews[0]:
            raise ValueError("Reviews require user IDs and valid timestamps")
        
    manifest = {
        'version': 1,
        'run': run.name,
        'counts': counts,
        'sources': {'business': str(business), 'review': str(reviews)},
    }
    write_json(run / 'manifest.json', manifest)
    write_json(root / 'active.json', manifest)
    return run


def existing_tables(cfg):
    root = cfg['data']['analytical_dir']
    manifest = json.loads((root / 'active.json').read_text(encoding='utf-8'))
    expected = {
        'business': str(cfg['data']['raw_dir'] / cfg['data']['business_file']),
        'review': str(cfg['data']['raw_dir'] / cfg['data']['review_file']),
    }
    if manifest['version'] != 1 or manifest['sources'] != expected:
        raise ValueError("Analytical inputs changed; rebuild preparation")
    
    run = root / manifest['run']
    for name in ('restaurants', 'reviews'):
        if not (run / f'{name}.parquet').is_file():
            raise FileNotFoundError(run / f'{name}.parquet')
        
    return run


def select_evidence(con, cfg, analytical, limit=None):
    """Apply serving filters without changing analytical tables."""
    serving = cfg['serving']
    con.execute(
        "CREATE VIEW core_restaurants AS SELECT * FROM read_parquet("
        + sql_string(analytical / 'restaurants.parquet')
        + ")"
    )
    cutoff = serving['before']
    where = "" if cutoff is None else " WHERE date < CAST(" + sql_string(cutoff) + " AS TIMESTAMP)"
    con.execute(
        "CREATE VIEW history AS SELECT * FROM read_parquet("
        + sql_string(analytical / "reviews.parquet")
        + ")"
        + where
    )
    if cutoff:
        if serving['open_only']:
            raise ValueError(
                "Historical builds require open_only: false; snapshot status leaks future information"
            )
        con.execute(
            "CREATE VIEW candidates AS SELECT b.* EXCLUDE(stars,review_count), h.stars, h.review_count FROM core_restaurants b JOIN (SELECT business_id, avg(stars) stars, count(*) review_count FROM history GROUP BY business_id) h USING(business_id)"
        )
    else:
        con.execute("CREATE VIEW candidates AS SELECT * FROM core_restaurants")
    conditions = [f"review_count >= {serving['min_review_count']}"]
    if serving['open_only']:
        conditions.append('is_open = 1')
    if serving['states']:
        conditions.append("state IN (" + ",".join(sql_string(s) for s in serving['states']) + ")")
    con.execute(
        "CREATE TABLE restaurants AS SELECT * FROM candidates WHERE " + " AND ".join(conditions)
    )
    cap = serving['reviews_per_restaurant']
    qualify = (
        ""
        if cap is None
        else f" QUALIFY row_number() OVER(PARTITION BY business_id ORDER BY date DESC, review_id) <= {cap}"
    )
    # Deterministic pseudo-random trial ordering avoids Parquet ordering bias
    end = ""
    if limit is not None:
        end = (
            " ORDER BY md5(review_id || "
            + sql_string(cfg['index']['seed'])
            + f"), review_id LIMIT {limit}"
        )
    con.execute(
        "CREATE VIEW evidence AS SELECT * FROM (SELECT h.* FROM history h SEMI JOIN restaurants USING(business_id) WHERE text IS NOT NULL AND length(trim(text))>0"
        + qualify
        + ")"
        + end
    )
