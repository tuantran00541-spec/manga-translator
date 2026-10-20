"""Writes the shop dataset (CSV files and schema.json) the hard probe runs on, the same every time."""
import csv
import json
import os
import random
import sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = random.Random(20261020)
FIRST = ["An", "Binh", "Chi", "Dung", "Giang", "Ha", "Hieu", "Khanh", "Lan", "Minh", "Nam", "Ngoc", "Phuong", "Quan", "Son", "Thao", "Trang", "Tuan", "Vy", "Yen"]
LAST = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Dang", "Bui", "Do", "Ho"]
CITIES = ["Hanoi", "Hue", "Da Nang", "Saigon", "Can Tho", "Hai Phong", None]
CATS = ["book", "game", "toy", "tool", "food", "music"]
SCHEMA = {
    "departments": [["id", "INTEGER"], ["name", "TEXT"], ["budget", "REAL"]],
    "employees": [["id", "INTEGER"], ["name", "TEXT"], ["dept_id", "INTEGER"], ["manager_id", "INTEGER"], ["salary", "INTEGER"], ["hired", "TEXT"]],
    "customers": [["id", "INTEGER"], ["name", "TEXT"], ["city", "TEXT"], ["tier", "INTEGER"], ["rep_id", "INTEGER"]],
    "products": [["id", "INTEGER"], ["name", "TEXT"], ["category", "TEXT"], ["price", "REAL"], ["stock", "INTEGER"]],
    "orders": [["id", "INTEGER"], ["customer_id", "INTEGER"], ["day", "TEXT"], ["status", "TEXT"], ["discount", "REAL"]],
    "order_items": [["order_id", "INTEGER"], ["product_id", "INTEGER"], ["qty", "INTEGER"], ["unit_price", "REAL"]],
    "payments": [["id", "INTEGER"], ["order_id", "INTEGER"], ["amount", "REAL"], ["method", "TEXT"]],
    "reviews": [["id", "INTEGER"], ["product_id", "INTEGER"], ["customer_id", "INTEGER"], ["stars", "INTEGER"], ["body", "TEXT"]],
}
rows = {name: [] for name in SCHEMA}
def maybe(value, p=0.08):
    return None if rng.random() < p else value


rows["departments"] = [[i, n, maybe(round(rng.uniform(5e4, 5e5), 2), 0.15)] for i, n in
                       enumerate(["Sales", "Support", "Warehouse", "Finance", "Research", "Legal", "Empty"], 1)]
for i in range(1, 121):
    rows["employees"].append([i, f"{rng.choice(FIRST)} {rng.choice(LAST)}", maybe(rng.randint(1, 6), 0.05),
                              None if i <= 3 else rng.randint(1, min(i - 1, 30)), rng.randrange(800, 9000, 50),
                              f"20{rng.randint(10, 25):02d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"])
for i in range(1, 601):
    rows["customers"].append([i, f"{rng.choice(LAST)} {rng.choice(FIRST)}", rng.choice(CITIES), maybe(rng.randint(1, 4)), maybe(rng.randint(1, 120), 0.2)])
for i in range(1, 181):
    cat = rng.choice(CATS)
    name = f"{cat}_{i}" + rng.choice(["", " deluxe", " 50%", " mini_box", "'s pick"])
    rows["products"].append([i, name, maybe(cat, 0.04), round(rng.choice([rng.uniform(1, 40), rng.uniform(40, 400)]), 2), maybe(rng.randint(0, 300))])
item_rows = []
for i in range(1, 2201):
    rows["orders"].append([i, maybe(rng.randint(1, 640), 0.02), f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                           rng.choices(["paid", "shipped", "cancelled", "refunded", "new"], [40, 30, 10, 5, 15])[0], maybe(rng.choice([0.0, 0.05, 0.1, 0.25]), 0.3)])
    for product in rng.sample(range(1, 186), rng.randint(1, 5)):
        item_rows.append([i, product, rng.randint(1, 9), maybe(round(rng.uniform(1, 400), 2), 0.03)])
rows["order_items"] = item_rows
pid = 0
for order in rows["orders"]:
    for _ in range(rng.choice([0, 1, 1, 1, 2])):
        pid += 1
        rows["payments"].append([pid, order[0], round(rng.uniform(-50, 900), 2), maybe(rng.choice(["card", "cash", "transfer", "wallet"]), 0.05)])
WORDS = ["great", "bad", "ok", "100%", "love_it", "meh", "broken", "fast", "slow", "GREAT", "50% off", "a_b"]
for i in range(1, 1501):
    rows["reviews"].append([i, rng.randint(1, 190), maybe(rng.randint(1, 600), 0.05), maybe(rng.randint(1, 5), 0.06),
                            maybe(" ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 4))), 0.1)])
for name, cols in SCHEMA.items():
    with open(os.path.join(out, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in cols])
        w.writerows([["" if v is None else v for v in row] for row in rows[name]])
json.dump(SCHEMA, open(os.path.join(out, "schema.json"), "w"), indent=1)
print({name: len(r) for name, r in rows.items()})
