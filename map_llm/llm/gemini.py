"""
Gemini LLM client: place type suggestion and optional LLM-based re-ranking.
All calls go through Vertex AI — no direct Gemini API key needed.
"""
import json
import time
from typing import List

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_random_exponential

from map_llm.config import cfg
from map_llm.models import Place, Recommendation
from map_llm.observability.logger import StructuredLogger

logger = StructuredLogger("llm.gemini")

_client: genai.Client | None = None


def init_vertex() -> None:
    global _client
    _client = genai.Client(vertexai=True, project=cfg.gcp.project, location=cfg.gcp.location)


def _get_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("Call init_vertex() before making LLM calls.")
    return _client


def _json_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=cfg.llm.temperature,
        response_mime_type="application/json",
    )


# Full list of supported Google Places API types:
# https://developers.google.com/maps/documentation/places/web-service/place-types
_AVAILABLE_TYPES = [
    # Food & drink
    "cafe", "coffee_shop", "restaurant", "bar", "bakery",
    "meal_takeaway", "fast_food_restaurant", "food_court",
    "ice_cream_shop", "juice_bar", "sandwich_shop",
    "wine_bar", "pub", "night_club", "pizza_restaurant",
    "sushi_restaurant", "ramen_restaurant", "hamburger_restaurant",
    # Activities
    "gym", "spa", "movie_theater", "bowling_alley",
    "museum", "art_gallery", "park", "tourist_attraction",
    # Accommodation
    "hotel", "motel", "hostel",
    # Shopping
    "shopping_mall", "book_store", "clothing_store",
]


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10), reraise=True)
def suggest_place_types(intent: str, request_id: str) -> list[str]:
    """
    Map the user's natural language place intent to a shortlist of valid
    Google Places API types. This is the only LLM call in the question phase —
    type mapping is semantic work that regex/lookup tables handle poorly
    (e.g. "boba shop" → ["cafe", "juice_bar"]).
    """
    types_list = ", ".join(_AVAILABLE_TYPES)

    prompt = f"""You are a Google Places API expert. Map the user's place intent to the most relevant Google Places API types.

User intent: "{intent}"

Available types: {types_list}

Return 4–6 types that best match the intent, ordered from most to least relevant.
Return JSON only:
{{
  "types": ["type1", "type2", ...]
}}"""

    logger.info(request_id, "llm_suggest_types_start", intent=intent, model=cfg.llm.model)
    t0 = time.time()

    response = _get_client().models.generate_content(
        model=cfg.llm.model, contents=prompt, config=_json_config()
    )
    latency_ms = round((time.time() - t0) * 1000, 1)

    result = json.loads(response.text)
    suggested = result["types"]
    logger.info(request_id, "llm_suggest_types_done", types=suggested, latency_ms=latency_ms)
    return suggested


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10), reraise=True)
def rerank_places(
    user_query: str,
    intent: str,
    selected_types: list[str],
    min_rating: float,
    max_rating: float,
    city: str,
    country: str,
    places: List[Place],
    top_k: int,
    request_id: str,
) -> List[Recommendation]:
    """
    Re-rank Nearby Search candidates against the full stated user intent.
    Not used in the main pipeline — embedding re-rank is faster and more consistent.
    Retained as an alternative for complex constraint queries that cosine similarity
    cannot capture (see docs/ml_methods.md — "What the Pipeline Does Not Do").
    """
    candidates_text = "\n".join(
        f"{i + 1}. {p.name} | Type: {p.primary_type or 'N/A'} | "
        f"Rating: {p.rating} ({p.user_rating_count} reviews) | "
        f"Price: {p.price_level or 'N/A'} | Open: {p.open_now} | {p.address}"
        for i, p in enumerate(places)
    )

    prompt = f"""You are a place recommendation expert. Re-rank these candidates by how well they match the user's requirements. Penalise places that clearly violate constraints.

Original query: "{user_query}"
Intent: {intent}
Types selected: {selected_types}
Rating range: {min_rating}–{max_rating}
Location: {city}, {country}

Candidates:
{candidates_text}

Return exactly {top_k} results as a JSON array (best match first):
[
  {{
    "rank": 1,
    "place_index": <1-based index>,
    "score": <0.0–1.0>,
    "reason": "<one sentence: which specific constraint this place best satisfies>"
  }}
]"""

    logger.info(
        request_id, "llm_rerank_start",
        candidate_count=len(places), top_k=top_k, model=cfg.llm.model,
    )
    t0 = time.time()

    response = _get_client().models.generate_content(
        model=cfg.llm.model, contents=prompt, config=_json_config()
    )
    latency_ms = round((time.time() - t0) * 1000, 1)

    rankings = json.loads(response.text)
    recommendations = [
        Recommendation(
            rank=r["rank"],
            place=places[r["place_index"] - 1],
            score=r["score"],
            reason=r["reason"],
        )
        for r in rankings[:top_k]
    ]

    logger.info(
        request_id, "llm_rerank_done",
        results=[rec.place.name for rec in recommendations], latency_ms=latency_ms,
    )
    return recommendations
