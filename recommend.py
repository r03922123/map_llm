#!/usr/bin/env python3
"""
LLM-powered place recommendation CLI.

Usage:
    python recommend.py "I want good ramen"
    python recommend.py "rooftop bar" --top-k 3 --radius 1500
"""
import argparse
import os
import sys
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from map_llm.config import cfg
from map_llm.llm.gemini import init_vertex, suggest_place_types
from map_llm.models import NearbySearchParams, UserIntent
from map_llm.observability.logger import StructuredLogger
from map_llm.pipeline.ranker import rank_by_similarity
from map_llm.pipeline.recall import geocode, nearby_search

logger = StructuredLogger("cli.recommend")

SEPARATOR = "─" * 50


def _ask_rating_bounds() -> tuple[float, float]:
    """Q1: structured rating range with explicit lower/upper bound."""
    print(f"\n{SEPARATOR}")
    print("Q1  Rating range  (scale 1.0 – 5.0)")
    print(SEPARATOR)

    while True:
        try:
            low  = input("  Lower bound (e.g. 4.0): ").strip() or "1.0"
            high = input("  Upper bound (e.g. 5.0): ").strip() or "5.0"
            lo, hi = float(low), float(high)
            if not (1.0 <= lo <= hi <= 5.0):
                print("  ! Lower must be ≤ Upper, both within 1.0–5.0. Try again.")
                continue
            return lo, hi
        except ValueError:
            print("  ! Enter a number like 3.5. Try again.")


def _ask_place_type(request_id: str) -> tuple[str, list[str]]:
    """
    Q2: two-step type selection.
      Step a — free text intent
      Step b — LLM maps to API types → user picks by number
    """
    print(f"\n{SEPARATOR}")
    print("Q2  Place type")
    print(SEPARATOR)

    intent_text = input("  What kind of place? (e.g. coffee shop, ramen, rooftop bar)\n  > ").strip()

    print("\n  Finding matching types...", flush=True)
    suggested = suggest_place_types(intent_text, request_id)

    print(f"\n  Available types for '{intent_text}':")
    for i, t in enumerate(suggested, 1):
        print(f"    {i}. {t}")

    while True:
        raw = input("\n  Select type(s) by number (e.g. 1  or  1,2): ").strip()
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            if all(1 <= i <= len(suggested) for i in indices):
                selected = [suggested[i - 1] for i in indices]
                return intent_text, selected
            print(f"  ! Enter numbers between 1 and {len(suggested)}.")
        except ValueError:
            print("  ! Enter numbers only, e.g. 1 or 1,3.")


def _ask_description() -> str:
    """Q4: free-text ideal place description — embedded for semantic similarity ranking."""
    print(f"\n{SEPARATOR}")
    print("Q4  Describe your ideal place")
    print(SEPARATOR)
    while True:
        desc = input(
            "  In 1–2 sentences, what are you looking for?\n"
            "  (e.g. 'Quiet corner spot, natural light, good for solo laptop work')\n"
            "  > "
        ).strip()
        if desc:
            return desc
        print("  ! Please enter a description.")


def _ask_geography() -> tuple[str, str]:
    """Q3: explicit city + country — split so geocoding gets a clean, unambiguous string."""
    print(f"\n{SEPARATOR}")
    print("Q3  Location")
    print(SEPARATOR)
    city    = input("  City:    ").strip()
    country = input("  Country: ").strip()
    return city, country


def _fmt_open(open_now) -> str:
    if open_now is True:  return "Open now"
    if open_now is False: return "Closed"
    return "Hours unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-powered place recommendations")
    parser.add_argument("query", help='e.g. "good ramen near me"')
    parser.add_argument("--top-k",  type=int, default=cfg.results.default_top_k,       help=f"Results to show (default: {cfg.results.default_top_k})")
    parser.add_argument("--radius", type=int, default=cfg.recall.default_radius_meters, help=f"Search radius in metres (default: {cfg.recall.default_radius_meters})")
    args = parser.parse_args()

    api_key = os.getenv("PLACES_API_KEY")
    if not api_key:
        sys.exit("Error: PLACES_API_KEY not set.")

    request_id = uuid.uuid4().hex[:8]
    logger.info(request_id, "session_start", query=args.query)
    t_start = time.time()

    init_vertex()

    print(f'\nQuery: "{args.query}"')

    # ── Structured question phase (deterministic — no LLM except Q2b type mapping) ──

    min_rating, max_rating = _ask_rating_bounds()
    logger.info(request_id, "q1_rating", min=min_rating, max=max_rating)

    intent_text, selected_types = _ask_place_type(request_id)
    logger.info(request_id, "q2_types", intent=intent_text, selected=selected_types)

    city, country = _ask_geography()
    logger.info(request_id, "q3_location", city=city, country=country)

    description = _ask_description()
    logger.info(request_id, "q4_description", description=description)

    user_intent = UserIntent(
        query=args.query,
        min_rating=min_rating,
        max_rating=max_rating,
        intent_text=intent_text,
        selected_types=selected_types,
        city=city,
        country=country,
        description=description,
    )

    # ── Geocode: location name → lat/lng ──────────────────────────────────────
    print(f"\n  Geocoding '{user_intent.location_name}'...")
    lat, lng = geocode(user_intent.location_name, api_key, request_id)

    params = NearbySearchParams(
        location_name=user_intent.location_name,
        latitude=lat,
        longitude=lng,
        radius_meters=args.radius,
        included_types=selected_types,
        keyword=None,
        max_results=cfg.places.max_candidates,
    )

    # ── Recall: Nearby Search API ─────────────────────────────────────────────
    print(f"  Searching nearby ({args.radius}m radius)...")
    candidates = nearby_search(params, api_key, request_id)

    if not candidates:
        print("No places found. Try --radius 2000 or a different location.")
        return

    # ── Filter: rating range (Nearby Search has no native rating filter) ──────
    rated = [p for p in candidates if p.rating is not None]
    in_range = [p for p in rated if min_rating <= p.rating <= max_rating]
    unrated  = [p for p in candidates if p.rating is None]
    filtered = in_range + unrated   # keep unrated — they may be new/good

    logger.info(
        request_id, "rating_filter",
        before=len(candidates), after=len(filtered),
        min=min_rating, max=max_rating,
    )

    if not filtered:
        print(f"No places found with rating {min_rating}–{max_rating}. Try widening the range.")
        return

    # ── Rank: embedding re-rank ───────────────────────────────────────────────
    print("  Ranking by semantic similarity to your description...")
    ranked = rank_by_similarity(
        description=user_intent.description,
        places=filtered,
        top_k=cfg.embedding.top_k,
        request_id=request_id,
    )

    total_ms = round((time.time() - t_start) * 1000, 1)
    logger.info(request_id, "request_complete", total_latency_ms=total_ms)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print(f"  Top {len(ranked)} of {len(filtered)}  |  {city}, {country}  |  rating {min_rating}–{max_rating}")
    print(f"  Types: {selected_types}")
    print(f"  Your description: \"{user_intent.description}\"")
    print(f"{'━' * 60}\n")

    for rank, (p, score) in enumerate(ranked, 1):
        stars   = f"{p.rating} ⭐" if p.rating else "No rating"
        n_rev   = f"({p.user_rating_count:,} reviews)" if p.user_rating_count else ""
        price   = p.price_level.replace("PRICE_LEVEL_", "").title() if p.price_level else "N/A"
        has_rev = f"  {len(p.reviews)} review snippets" if p.reviews else "  no reviews (name+address used)"

        print(f"#{rank}  {p.name}  [similarity: {score:.2%}]")
        print(f"    {p.address}")
        print(f"    {stars} {n_rev}  |  Price: {price}  |  {_fmt_open(p.open_now)}")
        print(f"   {has_rev}")
        if p.maps_url:
            print(f"    Maps: {p.maps_url}")
        print()

    print(f"  request_id={request_id}  total={total_ms}ms")


if __name__ == "__main__":
    main()
