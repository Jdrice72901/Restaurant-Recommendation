"""Write a small fake Yelp dataset in the real file format, for a practice run.

    python scripts/make_sample_data.py           # writes data/raw/yelp_academic_dataset_*.json
    python scripts/make_sample_data.py --force   # overwrite files that are already there

Then build and run as usual:

    python scripts/build_index.py
    streamlit run app.py

The restaurants and reviews are made up. When the real download is ready, copy
the real Yelp files over these two.
"""

import argparse
import json
import random
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Standard library only, so this runs before `pip install`. Matches the defaults in config.yaml.
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"

CITIES = {
    "PA": [("Philadelphia", 39.95, -75.16), ("Pittsburgh", 40.44, -79.99), ("Lancaster", 40.04, -76.31)],
    "FL": [("Tampa", 27.95, -82.46)],
}

# cuisine: (categories, price range, name parts, dishes)
CUISINES = {
    "Thai": ("Restaurants, Thai", (1, 2), ["Thai Orchid", "Bangkok Garden", "Siam Kitchen", "Lemongrass"],
             ["pad thai", "green curry", "drunken noodles", "tom yum soup", "mango sticky rice"]),
    "Italian": ("Restaurants, Italian, Pasta Shops", (2, 3), ["Trattoria Roma", "Nonna's Table", "Vino e Pasta"],
                ["handmade pappardelle", "carbonara", "osso buco", "tiramisu", "margherita pizza"]),
    "Pizza": ("Pizza, Restaurants", (1, 2), ["Slice Palace", "Brick Oven Co.", "Pie Society"],
              ["pepperoni slice", "white pie", "garlic knots", "grandma pizza", "buffalo wings"]),
    "Mexican": ("Mexican, Restaurants, Tacos", (1, 2), ["Taqueria El Sol", "Casa Verde", "La Cocina"],
                ["al pastor tacos", "carnitas", "elote", "fresh guacamole", "churros"]),
    "Japanese": ("Japanese, Sushi Bars, Ramen, Restaurants", (2, 3), ["Ramen Ya", "Sakura Sushi", "Izakaya Kumo"],
                 ["tonkotsu ramen", "spicy tuna roll", "omakase", "gyoza", "karaage"]),
    "Indian": ("Indian, Restaurants", (1, 2), ["Taj Spice", "Curry House", "Masala Corner"],
               ["butter chicken", "lamb vindaloo", "garlic naan", "saag paneer", "chicken biryani"]),
    "Vegan": ("Vegan, Vegetarian, Restaurants", (2,), ["Green Leaf", "Root & Sprout"],
              ["jackfruit tacos", "cauliflower wings", "grain bowl", "cashew mac and cheese", "beet burger"]),
    "Steakhouse": ("Steakhouses, Restaurants, American (New)", (3, 4), ["Prime Cut", "The Butcher's Grill"],
                   ["dry-aged ribeye", "filet mignon", "creamed spinach", "wedge salad", "lobster tail"]),
    "Breakfast": ("Breakfast & Brunch, Restaurants, Cafes", (1, 2), ["Sunny Side Diner", "The Early Bird"],
                  ["buttermilk pancakes", "eggs benedict", "breakfast burrito", "french toast", "home fries"]),
    "BBQ": ("Barbeque, Restaurants, Southern", (2,), ["Smokehouse 54", "Low & Slow BBQ"],
            ["brisket", "pulled pork", "burnt ends", "mac and cheese", "cornbread"]),
}

POSITIVE = [
    "The {dish} was incredible, easily the best I've had in {city}.",
    "We came for the {dish} and it did not disappoint. Huge portions too.",
    "Cozy little spot with friendly staff. The {dish} and the {dish2} were both great.",
    "Service was quick and the {dish} came out hot. Will be back.",
    "Perfect for a date night. Dim lighting, good music, and amazing {dish}.",
    "Great value for the price. The {dish} is a must order.",
    "Came with a big group and they handled us well. Everyone loved the {dish}.",
    "Quiet, relaxed atmosphere, good for working or a long lunch. Try the {dish}.",
    "Kid friendly and the staff were patient with our toddler. The {dish} was solid.",
    "Authentic flavors, you can tell they care. The {dish2} was a nice surprise.",
]
NEGATIVE = [
    "The {dish} was bland and the wait was over an hour.",
    "Overpriced for what you get. The {dish} was cold when it arrived.",
    "Loud and crowded, hard to have a conversation. {dish} was just okay.",
    "Rude server and the {dish} tasted reheated. Not coming back.",
]
EXTRAS = [
    "Parking can be tricky on weekends.",
    "They have a nice outdoor patio.",
    "Reservations recommended on Friday nights.",
    "Takeout was fast and well packaged.",
    "Vegetarian options are limited.",
    "Lots of vegetarian and vegan options.",
    "The cocktails are worth trying.",
    "",
]

NON_RESTAURANTS = [
    ("Ace Hardware", "Hardware Stores, Home & Garden, Shopping"),
    ("Main Street Dental", "Dentists, Health & Medical"),
    ("Downtown Dry Cleaning", "Dry Cleaning & Laundry, Local Services"),
]


def yelp_id(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_letters + string.digits + "-_", k=22))


def make_business(rng, name, city, state, lat, lon, categories, stars, review_count, is_open, price):
    attrs = {"RestaurantsPriceRange2": str(price), "WiFi": "u'free'"} if price else None
    return {
        "business_id": yelp_id(rng),
        "name": name,
        "address": f"{rng.randint(10, 9999)} {rng.choice(['Main', 'Market', 'Walnut', 'Chestnut', 'Penn', 'Oak'])} St",
        "city": city,
        "state": state,
        "postal_code": f"{rng.randint(10000, 99999)}",
        "latitude": round(lat + rng.uniform(-0.05, 0.05), 6),
        "longitude": round(lon + rng.uniform(-0.05, 0.05), 6),
        "stars": stars,
        "review_count": review_count,
        "is_open": is_open,
        "attributes": attrs,
        "categories": categories,
        "hours": {"Monday": "11:0-22:0", "Friday": "11:0-23:0", "Saturday": "10:0-23:0"},
    }


def make_review(rng, business_id, text, stars, start):
    return {
        "review_id": yelp_id(rng),
        "user_id": yelp_id(rng),
        "business_id": business_id,
        "stars": float(stars),
        "useful": rng.randint(0, 25),
        "funny": rng.randint(0, 5),
        "cool": rng.randint(0, 8),
        "text": text,
        "date": (start + timedelta(days=rng.randint(0, 1800), seconds=rng.randint(0, 86399))).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


def generate(seed: int = 42):
    rng = random.Random(seed)
    start = datetime(2018, 1, 1)
    businesses, reviews = [], []

    for state, cities in CITIES.items():
        for city, lat, lon in cities:
            for cuisine, (categories, prices, names, dishes) in CUISINES.items():
                name = rng.choice(names)
                stars = rng.choice([2.5, 3.0, 3.5, 3.5, 4.0, 4.0, 4.5, 4.5, 5.0])
                n_reviews = rng.randint(12, 30)
                is_open = 0 if rng.random() < 0.1 else 1
                b = make_business(rng, name, city, state, lat, lon, categories, stars,
                                  rng.randint(n_reviews, 900), is_open, rng.choice(prices))
                businesses.append(b)

                for _ in range(n_reviews):
                    good = rng.random() < (stars - 1) / 4
                    template = rng.choice(POSITIVE if good else NEGATIVE)
                    d1, d2 = rng.sample(dishes, 2)
                    text = (template.format(dish=d1, dish2=d2, city=city) + " " + rng.choice(EXTRAS)).strip()
                    rating = rng.choice([4, 5]) if good else rng.choice([1, 2, 3])
                    reviews.append(make_review(rng, b["business_id"], text, rating, start))

            for name, categories in NON_RESTAURANTS:
                b = make_business(rng, name, city, state, lat, lon, categories, 4.0, 25, 1, None)
                businesses.append(b)
                reviews.append(make_review(rng, b["business_id"], "Helpful staff, found what I needed.", 4, start))

    return businesses, reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite existing files in data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = RAW_DIR
    business_path = raw_dir / BUSINESS_FILE
    review_path = raw_dir / REVIEW_FILE
    existing = [p for p in (business_path, review_path) if p.exists()]
    if existing and not args.force:
        sys.exit(f"{existing[0]} already exists. Pass --force to overwrite it with sample data.")

    businesses, reviews = generate(args.seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(business_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(b) + "\n" for b in businesses)
    with open(review_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in reviews)
    print(f"Wrote {len(businesses)} businesses and {len(reviews)} reviews to {raw_dir}")


if __name__ == "__main__":
    main()
