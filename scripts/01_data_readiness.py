"""Finalize Philadelphia data inputs without embedding or scoring holdouts.

Consumes the audited prep_eda exports. Run from any directory using this script's path.
Review text stays in the immutable analytical core; derived membership tables reference
it by review_id, avoiding another copy of the entire review corpus.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / 'config_project.yaml').read_text(encoding='utf-8'))
    analytical = ROOT / 'data/processed/analytical'
    manifest = json.loads((analytical / 'active.json').read_text(encoding='utf-8'))
    core = analytical / manifest['run']
    eda = ROOT / 'data/processed/eda' / manifest['run']
    out = ROOT / 'data/processed/philadelphia' / manifest['run']
    out.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(eda / 'restaurant_snapshot_features.parquet')
    flags = pd.read_parquet(eda / 'followup/geographic_flags.parquet')
    assert features.business_id.is_unique and flags.business_id.is_unique
    scoped = features.loc[
        features.city_normalized.eq(cfg['scope']['city_normalized'])
        & features.state_normalized.eq(cfg['scope']['state_normalized'])
    ].copy()
    scoped = scoped.merge(
        flags[['business_id', 'geo_review_needed']], on='business_id', validate='one_to_one'
    )
    assert scoped.geo_review_needed.notna().all()
    scoped['index_geography_eligible'] = scoped.coordinates_valid
    if cfg['scope']['exclude_flagged_coordinates']:
        scoped['index_geography_eligible'] &= ~scoped.geo_review_needed
    scoped['serving_eligible'] = scoped.index_geography_eligible
    if cfg['scope']['serving_snapshot_open_only']:
        scoped['serving_eligible'] &= scoped.is_open.eq(1)
    # Unknown amenities remain nullable. Never manufacture a negative or affordable price.
    assert scoped.price.dropna().isin([1, 2, 3, 4]).all()
    for column in ['delivery', 'takeout', 'outdoor_seating']:
        scoped[column] = scoped[column].astype('boolean')
    scoped.to_parquet(out / 'restaurant_features.parquet', index=False)
    columns = [
        'business_id',
        'city_normalized',
        'state_normalized',
        'price_known',
        'coordinates_valid',
        'delivery',
        'takeout',
        'outdoor_seating',
        'geo_review_needed',
        'index_geography_eligible',
        'serving_eligible',
    ]
    with duckdb.connect() as con:
        con.read_parquet(str(core / 'restaurants.parquet')).create_view('raw_restaurants')
        con.read_parquet(str(core / 'reviews.parquet')).create_view('source_reviews')
        con.register('normalized', scoped[columns])
        con.execute(
            'CREATE TABLE restaurants AS SELECT b.*, n.* EXCLUDE(business_id) '
            'FROM raw_restaurants b JOIN normalized n USING(business_id)'
        )
        con.execute(
            'CREATE TABLE reviews AS SELECT r.* FROM source_reviews r '
            'SEMI JOIN restaurants USING(business_id)'
        )
        train_end = cfg['evaluation']['train_end']
        validation_end = cfg['evaluation']['validation_end']
        con.execute(
            '''CREATE TABLE membership AS
            SELECT review_id,user_id,business_id,date,
                CASE WHEN date < CAST(? AS TIMESTAMP) THEN 'train'
                     WHEN date < CAST(? AS TIMESTAMP) THEN 'validation' ELSE 'test' END split
            FROM reviews''',
            [train_end, validation_end],
        )
        con.execute('''CREATE TABLE train_pairs AS
            SELECT r.user_id,r.business_id,r.review_id latest_review_id,r.date latest_date,
                   r.stars latest_stars,
                   count(*) OVER(PARTITION BY r.user_id,r.business_id) interaction_count
            FROM reviews r JOIN membership m USING(review_id)
            WHERE m.split='train'
            QUALIFY row_number() OVER(PARTITION BY r.user_id,r.business_id
                                     ORDER BY r.date DESC,r.review_id DESC)=1''')
        con.execute('''CREATE TABLE train_item_features AS
            SELECT business_id,count(*) train_reviews,count(DISTINCT user_id) train_users,
                   avg(stars) train_mean_stars
            FROM reviews JOIN membership USING(review_id,user_id,business_id,date) WHERE split='train'
            GROUP BY business_id''')
        con.execute('''CREATE TABLE train_user_features AS
            SELECT user_id,count(*) train_reviews,count(DISTINCT business_id) train_restaurants
            FROM reviews JOIN membership USING(review_id,user_id,business_id,date) WHERE split='train' GROUP BY user_id''')
        con.execute('''CREATE TABLE index_membership AS
            SELECT m.review_id,m.business_id,m.split,
                   b.index_geography_eligible AND m.split='train' evaluation_evidence,
                   b.serving_eligible serving_evidence
            FROM membership m JOIN restaurants b USING(business_id)''')
        for table, filename in [
            ('restaurants', 'restaurants'),
            ('membership', 'split_membership'),
            ('train_pairs', 'train_pairs'),
            ('train_item_features', 'train_item_features'),
            ('train_user_features', 'train_user_features'),
            ('index_membership', 'index_membership'),
        ]:
            con.table(table).write_parquet(str(out / f'{filename}.parquet'), compression='zstd')
        con.read_parquet(str(eda / 'category_labels.parquet')).create_view('all_categories')
        con.sql(
            'SELECT c.* FROM all_categories c SEMI JOIN restaurants USING(business_id)'
        ).write_parquet(str(out / 'category_labels.parquet'), compression='zstd')
        splits = con.sql('''SELECT split,count(*) reviews,count(DISTINCT user_id) users,
            count(DISTINCT business_id) restaurants FROM membership GROUP BY split''').df()
        splits.to_csv(out / 'split_summary.csv', index=False)
        counts = (
            con.sql('''SELECT count(*) scope_reviews,
            count(*) FILTER(WHERE evaluation_evidence) evaluation_evidence_reviews,
            count(*) FILTER(WHERE serving_evidence) serving_evidence_reviews
            FROM index_membership''')
            .df()
            .iloc[0]
            .to_dict()
        )
        assert con.sql('SELECT count(*)=count(DISTINCT review_id) FROM membership').fetchone()[0]
        assert con.sql('SELECT sum(interaction_count) FROM train_pairs').fetchone()[0] == int(
            splits.loc[splits.split.eq('train'), 'reviews'].iloc[0]
        )
        assert con.sql('SELECT max(latest_date) FROM train_pairs').fetchone()[0] < pd.Timestamp(
            train_end
        )
        counts.update(
            scope_restaurants=len(scoped),
            geography_excluded=(~scoped.index_geography_eligible).sum(),
            serving_restaurants=int(scoped.serving_eligible.sum()),
        )
    report = {
        'analytical_run': manifest['run'],
        'decisions': cfg,
        'counts': counts,
        'review_source': str((core / 'reviews.parquet').relative_to(ROOT)),
        'notes': 'Historical snapshot metadata; no holdout scores or embeddings created.',
    }
    (out / 'readiness.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(counts, indent=2))
    print(splits.to_string(index=False))
    print(f'Finalized data inputs: {out}')


if __name__ == '__main__':
    main()
