from config import CATEGORIES, STATUSES
from dataset import ORDERS, category_counts, delayed_percentage, status_counts


def test_order_count():
    assert len(ORDERS) >= 40


def test_every_category():
    counts = category_counts()
    for cat in CATEGORIES:
        assert counts.get(cat, 0) >= 3


def test_every_status():
    counts = status_counts()
    for st in STATUSES:
        assert counts.get(st, 0) >= 1


def test_delayed_band():
    pct = delayed_percentage()
    assert 10 <= pct <= 30


def test_order_fields():
    for o in ORDERS:
        assert set(o) >= {
            "record_id",
            "category",
            "status",
            "order_value_inr",
            "days_since_created",
            "delayed_shipment",
        }
        assert 0 <= o["days_since_created"] <= 30
        assert isinstance(o["delayed_shipment"], bool)
