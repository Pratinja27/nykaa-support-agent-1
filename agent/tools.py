from config import DELAYED_WEIGHT, ESCALATION_THRESHOLD, RECENCY_WEIGHT
from dataset import ORDERS

ORDER_INDEX = {o["record_id"]: o for o in ORDERS}


def escalation_score(order):
    delayed = 1.0 if order["delayed_shipment"] else 0.0
    recency = order["days_since_created"] / 30.0
    score = DELAYED_WEIGHT * delayed + RECENCY_WEIGHT * recency
    if score < 0:
        score = 0.0
    if score > 1:
        score = 1.0
    return round(score, 4)


def check_order_status(record_id):
    key = (record_id or "").strip().upper()
    order = ORDER_INDEX.get(key)
    if not order:
        return {
            "found": False,
            "record_id": key,
            "status": None,
            "order_value_inr": None,
            "escalation_score": None,
            "escalate": False,
        }
    score = escalation_score(order)
    return {
        "found": True,
        "record_id": order["record_id"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "category": order["category"],
        "days_since_created": order["days_since_created"],
        "delayed_shipment": order["delayed_shipment"],
        "escalation_score": score,
        "escalate": score >= ESCALATION_THRESHOLD,
    }
