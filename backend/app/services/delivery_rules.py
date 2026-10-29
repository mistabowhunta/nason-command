import json
import os
import threading

RULES_FILE = "/app/app/known_faces/delivery_rules.json"
_lock = threading.Lock()

DEFAULT_RULES = [
    {
        "id": "AMAZON",
        "name": "AMAZON",
        "kind": "carrier",
        "alert_mode": "alert",
        "cues": "Amazon Prime logo, blue delivery vest, Amazon delivery van/Rivian, cardboard packages/totes",
        "sightings": 0,
        "last_seen": None,
    },
    {
        "id": "UPS",
        "name": "UPS",
        "kind": "carrier",
        "alert_mode": "alert",
        "cues": "UPS shield logo, brown delivery truck/package car, brown uniform, handheld DIAD scanner",
        "sightings": 0,
        "last_seen": None,
    },
    {
        "id": "FEDEX",
        "name": "FEDEX",
        "kind": "carrier",
        "alert_mode": "alert",
        "cues": "FedEx purple/orange logo, white/dark delivery van, FedEx uniform shirt/vest, parcels",
        "sightings": 0,
        "last_seen": None,
    },
    {
        "id": "USPS",
        "name": "USPS",
        "kind": "carrier",
        "alert_mode": "alert",
        "cues": "USPS eagle logo, white postal mail truck, blue postal carrier uniform/satchel",
        "sightings": 0,
        "last_seen": None,
    },
    {
        "id": "LANDSCAPERS",
        "name": "LANDSCAPERS",
        "kind": "service",
        "alert_mode": "silent",
        "cues": "Landscaping/yard crew, lawn mower, leaf blower, string trimmer/edger, high-vis safety vest, bandana/work hat, utility trailer",
        "sightings": 0,
        "last_seen": None,
    },
]


def get_delivery_rules() -> list[dict]:
    with _lock:
        if not os.path.exists(RULES_FILE):
            os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
            with open(RULES_FILE, "w") as f:
                json.dump(DEFAULT_RULES, f, indent=2)
            return list(DEFAULT_RULES)
        try:
            with open(RULES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return list(DEFAULT_RULES)


def save_delivery_rules(rules: list[dict]) -> list[dict]:
    with _lock:
        os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
        with open(RULES_FILE, "w") as f:
            json.dump(rules, f, indent=2)
        return rules


def upsert_delivery_rule(name: str, kind: str = "carrier", alert_mode: str = "alert", cues: str = "") -> list[dict]:
    clean_name = name.strip().upper()
    rules = get_delivery_rules()
    found = False
    for r in rules:
        if r["id"] == clean_name:
            r["kind"] = kind
            r["alert_mode"] = alert_mode
            if cues:
                r["cues"] = cues
            found = True
            break
    if not found:
        rules.append({
            "id": clean_name,
            "name": clean_name,
            "kind": kind,
            "alert_mode": alert_mode,
            "cues": cues or f"{clean_name} logo, uniform, or service vehicle",
            "sightings": 0,
            "last_seen": None,
        })
    return save_delivery_rules(rules)


def delete_delivery_rule(rule_id: str) -> list[dict]:
    clean_id = rule_id.strip().upper()
    rules = [r for r in get_delivery_rules() if r["id"] != clean_id]
    return save_delivery_rules(rules)


def record_delivery_sighting(rule_id: str, iso_ts: str) -> dict | None:
    clean_id = rule_id.strip().upper()
    rules = get_delivery_rules()
    matched = None
    for r in rules:
        if r["id"] == clean_id:
            r["sightings"] = int(r.get("sightings") or 0) + 1
            r["last_seen"] = iso_ts
            matched = dict(r)
            break
    if matched:
        save_delivery_rules(rules)
    return matched
