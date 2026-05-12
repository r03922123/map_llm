import json
import time
from typing import List

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .logger import StructuredLogger
from .models import Place, Recommendation, SearchParams

logger = StructuredLogger("llm")

_MODEL = "publishers/google/models/gemini-2.5-flash"
_client: genai.Client | None = None


def init_vertex(project: str, location: str = "us-central1") -> None:
    global _client
    _client = genai.Client(vertexai=True, project=project, location=location)


def _get_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("Call init_vertex() before making LLM calls.")
    return _client


def _json_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
    )


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10), reraise=True)
def parse_query(user_query: str, request_id: str) -> SearchParams:
    """
    Turn the user's free-text query into an optimised Places API search string.
    Doing this with an LLM instead of regex lets it handle paraphrases,
    abbreviations, and implied constraints ("cheap but cozy" → price level signals).
    """
    prompt = f"""You are a Google Maps search expert. Convert the user's natural language query into the best possible Google Places text search string for coffee shops.

User query: "{user_query}"

Rules:
- The search_query must work well as a Google Places API textQuery
- Include relevant qualifiers the user implied (e.g. "quiet" → add "quiet cozy")
- Always include "coffee shop" or "cafe" in the search_query

Respond with JSON only:
{{
  "search_query": "<optimised Places API search string>",
  "reasoning": "<one sentence explaining the key translation choices>"
}}"""

    logger.info(request_id, "llm_parse_start", user_query=user_query, model=_MODEL)
    t0 = time.time()

    response = _get_client().models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=_json_config(),
    )
    latency_ms = round((time.time() - t0) * 1000, 1)

    result = json.loads(response.text)
    logger.info(request_id, "llm_parse_done", output=result, latency_ms=latency_ms)
    return SearchParams(**result)


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10), reraise=True)
def rerank_places(
    user_query: str,
    places: List[Place],
    top_k: int,
    request_id: str,
) -> List[Recommendation]:
    """
    Re-rank candidates against the original user intent.
    The Places API returns results by Google's default relevance, which doesn't
    know the user's specific constraints (e.g. "good wifi", "dog-friendly").
    LLM re-ranking closes that gap without a fine-tuned model.
    """
    candidates_text = "\n".join(
        f"{i + 1}. {p.name} | Rating: {p.rating} ({p.user_rating_count} reviews) "
        f"| Price: {p.price_level or 'N/A'} | Open now: {p.open_now} | {p.address}"
        for i, p in enumerate(places)
    )

    prompt = f"""You are a coffee shop recommendation expert. Re-rank the candidates below by how well they match the user's specific request. Be strict — penalise places that clearly don't fit.

User query: "{user_query}"

Candidates:
{candidates_text}

Return exactly {top_k} results as a JSON array (best match first):
[
  {{
    "rank": 1,
    "place_index": <1-based index from the candidates list>,
    "score": <0.0–1.0 match score>,
    "reason": "<one sentence: which specific aspect of the user query this place satisfies>"
  }}
]"""

    logger.info(
        request_id,
        "llm_rerank_start",
        user_query=user_query,
        candidate_count=len(places),
        top_k=top_k,
        model=_MODEL,
    )
    t0 = time.time()

    response = _get_client().models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=_json_config(),
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
        request_id,
        "llm_rerank_done",
        results=[rec.place.name for rec in recommendations],
        latency_ms=latency_ms,
    )
    return recommendations
