from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from recommender import index_hybrid

ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_app_filters_order_and_sources(monkeypatch):
    calls = []
    source = {
        'passage_id': 'review-a:0',
        'review_id': 'review-a',
        'date': '2019-01-01',
        'review_stars': 4.0,
        'text': 'Vegetarian dinner with outdoor seating'
    }

    class Index:
        def __init__(self, *args, **kwargs):
            self.catalog = pd.DataFrame([
                {
                    'business_id': bid,
                    'name': name,
                    'address': '123 Main St',
                    'city': 'Philadelphia',
                    'state': 'PA',
                    'price': None,
                    'stars': stars,
                    'review_count': 10,
                    'categories': 'Restaurants',
                    'latitude': 39.95,
                    'longitude': -75.16
                }
                for bid, name, stars in [('a', 'First By Retrieval', 3), ('b', 'Second', 5)]
            ])

        def search(self, query, **filters):
            calls.append((query, filters))
            return [
                {'business_id': bid, 'score': 0.03, 'evidence': [source, source]}
                for bid in ['a', 'b']
            ]

    monkeypatch.setattr(index_hybrid, 'GemmaEmbedder', lambda *args, **kwargs: object())
    monkeypatch.setattr(index_hybrid, 'HybridIndex', Index)
    st.cache_resource.clear()
    app = AppTest.from_file(str(ROOT / 'app_hybrid.py')).run()
    assert not app.exception
    app.text_input[0].set_value('vegetarian dinner')
    app.checkbox[0].check()
    app.checkbox[2].check()
    app.button[0].click().run()
    assert not app.exception
    assert calls == [(
        'vegetarian dinner',
        {'max_price': 4, 'delivery': False, 'takeout': True, 'outdoor': False}
    )]
    titles = [item.value for item in app.markdown if item.value.startswith('### ')]
    assert titles == ['### First By Retrieval', '### Second']
    assert len(app.expander) == 2
    assert app.text[0].value == source['text']
    assert app.selectbox[0].value == 'none'
    st.cache_resource.clear()
