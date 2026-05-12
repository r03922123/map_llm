#!/usr/bin/env python3
"""
Post-deploy smoke test. Exits 0 on pass, 1 on any failure.
Run after every deploy before marking the rollout complete (CLAUDE.md deployment gate).

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py https://map-llm-xxx.us-central1.run.app
"""
import sys
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://map-llm-1074296379160.us-central1.run.app"
TIMEOUT = 30

_TEST_PAYLOAD = {
    "query": "smoke test ramen",
    "min_rating": 1.0,
    "max_rating": 5.0,
    "intent_text": "restaurant",
    "selected_types": ["restaurant"],
    "city": "San Francisco",
    "country": "USA",
    "description": "any restaurant",
}


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{': ' + detail if detail else ''}")
        sys.exit(1)


def main():
    print(f"Smoke test → {BASE_URL}\n")

    # /health
    r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    check("/health status 200", r.status_code == 200, str(r.status_code))
    body = r.json()
    check("/health status=ok", body.get("status") == "ok", str(body))
    check("/health has variant_id", "variant_id" in body)
    check("/health has revision",   "revision"   in body)
    print(f"         variant_id={body['variant_id']}  revision={body['revision']}")

    # /suggest-types
    r = requests.get(f"{BASE_URL}/suggest-types", params={"intent": "ramen"}, timeout=TIMEOUT)
    check("/suggest-types status 200", r.status_code == 200, str(r.status_code))
    check("/suggest-types returns types", len(r.json().get("types", [])) > 0)

    # /recommend
    r = requests.post(f"{BASE_URL}/recommend", json=_TEST_PAYLOAD, timeout=TIMEOUT)
    check("/recommend status 200", r.status_code == 200, r.text[:200])
    body = r.json()
    check("/recommend has request_id",    "request_id"    in body)
    check("/recommend has variant_id",    "variant_id"    in body)
    check("/recommend has results",       len(body.get("results", [])) > 0)
    check("/recommend has candidate_count", "candidate_count" in body)

    print(f"\nAll checks passed. Service is healthy.")


if __name__ == "__main__":
    main()
