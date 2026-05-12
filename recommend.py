#!/usr/bin/env python3
"""
Coffee shop recommendation CLI.

Usage:
    python recommend.py "quiet cafe with good wifi near downtown SF" --top-k 5
    python recommend.py "cheap espresso bar in Brooklyn" --top-k 3 --candidates 30
"""
import argparse
import os
import sys
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from src.llm import init_vertex, parse_query, rerank_places
from src.logger import StructuredLogger
from src.places import search_places

logger = StructuredLogger("recommend")


def _fmt_open(open_now) -> str:
    if open_now is True:
        return "Open now"
    if open_now is False:
        return "Closed"
    return "Hours unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-powered coffee shop recommendations via Google Maps"
    )
    parser.add_argument("query", help='e.g. "quiet cafe with wifi near downtown SF"')
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="Places API candidate pool size before re-ranking (default: 20)",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)",
    )
    args = parser.parse_args()

    api_key = os.getenv("PLACES_API_KEY")
    if not api_key:
        sys.exit("Error: PLACES_API_KEY not set. Add it to .env or export it.")
    if not args.project:
        sys.exit("Error: GOOGLE_CLOUD_PROJECT not set. Add it to .env or pass --project.")

    request_id = uuid.uuid4().hex[:8]
    logger.info(request_id, "request_start", query=args.query, top_k=args.top_k)
    t_start = time.time()

    # Step 1 — LLM translates natural language → optimised Places search string
    init_vertex(project=args.project)
    search_params = parse_query(args.query, request_id)

    # Step 2 — Fetch candidate places from Google Maps
    candidates = search_places(
        query=search_params.search_query,
        api_key=api_key,
        max_results=args.candidates,
        request_id=request_id,
    )

    if not candidates:
        print("No places found. Try broadening your query.")
        return

    # Step 3 — LLM re-ranks candidates against the original user intent
    top_k = min(args.top_k, len(candidates))
    recommendations = rerank_places(
        user_query=args.query,
        places=candidates,
        top_k=top_k,
        request_id=request_id,
    )

    total_ms = round((time.time() - t_start) * 1000, 1)
    logger.info(request_id, "request_complete", total_latency_ms=total_ms)

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print(f"  Top {top_k} coffee shops for: \"{args.query}\"")
    print(f"{'━' * 60}\n")

    for rec in recommendations:
        p = rec.place
        stars = f"{p.rating} ⭐" if p.rating else "No rating"
        reviews = f"({p.user_rating_count:,} reviews)" if p.user_rating_count else ""
        price = p.price_level.replace("PRICE_LEVEL_", "").title() if p.price_level else "N/A"

        print(f"#{rec.rank}  {p.name}  [match: {rec.score:.0%}]")
        print(f"    {p.address}")
        print(f"    {stars} {reviews}  |  Price: {price}  |  {_fmt_open(p.open_now)}")
        print(f"    Why: {rec.reason}")
        if p.maps_url:
            print(f"    Maps: {p.maps_url}")
        print()

    print(f"  request_id={request_id}  total={total_ms}ms")


if __name__ == "__main__":
    main()
