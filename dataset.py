import random
from collections import Counter

from config import (
    CATEGORIES,
    CATEGORY_WEIGHTS,
    DELAYED_PROB,
    MAX_AMOUNT,
    MIN_AMOUNT,
    N_ORDERS,
    SEED,
    STATUSES,
    STATUS_WEIGHTS,
)

rng = random.Random(SEED)


def _amount_for(category):
    bands = {
        "Beauty": (199, 4999),
        "Apparel": (499, 7999),
        "Footwear": (799, 8999),
        "Home": (399, 9999),
        "Electronics": (999, 14999),
    }
    low, high = bands[category]
    low = max(low, MIN_AMOUNT)
    high = min(high, MAX_AMOUNT)
    return rng.randint(low, high)


def generate_orders():
    orders = []
    for i in range(N_ORDERS):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        days = rng.randint(0, 30)
        delayed = rng.random() < DELAYED_PROB
        orders.append(
            {
                "record_id": "NYK-%04d" % (1001 + i),
                "category": category,
                "status": status,
                "order_value_inr": _amount_for(category),
                "days_since_created": days,
                "delayed_shipment": delayed,
            }
        )
    return orders


ORDERS = generate_orders()


def category_counts(orders=None):
    rows = orders if orders is not None else ORDERS
    return dict(Counter(o["category"] for o in rows))


def status_counts(orders=None):
    rows = orders if orders is not None else ORDERS
    return dict(Counter(o["status"] for o in rows))


def delayed_percentage(orders=None):
    rows = orders if orders is not None else ORDERS
    n = sum(1 for o in rows if o["delayed_shipment"])
    return 100.0 * n / len(rows)


def print_report():
    cats = category_counts()
    stats = status_counts()
    delayed = delayed_percentage()
    print("seed:", SEED)
    print("orders:", len(ORDERS))
    print("category counts:", cats)
    print("status counts:", stats)
    print("delayed shipment percentage: %.2f%%" % delayed)
    print("amount range: INR %d to %d" % (MIN_AMOUNT, MAX_AMOUNT))
    print(
        "price range reasoning: Nykaa mixes drugstore beauty, apparel and a smaller electronics/home range, so 199-14999 INR covers everyday SKUs without luxury outliers."
    )
    if delayed < 10 or delayed > 30:
        raise SystemExit("delayed percentage outside 10-30, change seed or weights and regenerate")
    for cat in CATEGORIES:
        if cats.get(cat, 0) < 3:
            raise SystemExit("category %s has fewer than 3 records" % cat)
    for st in STATUSES:
        if stats.get(st, 0) < 1:
            raise SystemExit("status %s missing" % st)


if __name__ == "__main__":
    print_report()
    print("sample:", ORDERS[0])
