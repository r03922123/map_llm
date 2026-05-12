import time
from typing import List

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config import cfg
from .logger import StructuredLogger
from .models import NearbySearchParams, Place

logger = StructuredLogger("places")

# ── Field mask ────────────────────────────────────────────────────────────────
# SKU breakdown (billed at the highest tier present):
#   Basic    : displayName, formattedAddress, googleMapsUri, primaryType
#   Advanced : rating, userRatingCount, priceLevel, currentOpeningHours
#   Preferred: reviews                                ← drives SKU to $0.040/req
#
# No redundant fields found — all are used in rating filter, embedding text, or display.
#
# TODO: Text Search API differentiation
#   Nearby Search returns results sorted by proximity only — no keyword/semantic signal.
#   Text Search API (places:searchText) supports a `textQuery` field that does keyword
#   matching and returns up to 60 results with pagination via nextPageToken.
#   Use Text Search as a fallback when Nearby Search returns < N results, or when the
#   query has strong keyword intent (e.g. "best tonkotsu ramen").
#
# TODO: photo → vision text
#   places.photos returns photo references (not image bytes). To use visual signal:
#   1. GET /v1/{photo_reference}/media → raw image
#   2. Gemini Vision: "Describe this venue's atmosphere in 2 sentences"
#   3. Append vision text to place_text before embedding
#   This differentiates from Text Search which has no visual signal at all.
_NEARBY_FIELD_MASK = ",".join([
    "places.displayName",
    "places.formattedAddress",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.googleMapsUri",
    "places.currentOpeningHours",
    "places.primaryType",
    "places.reviews",
])


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10), reraise=True)
def geocode(location_name: str, api_key: str, request_id: str = "") -> tuple[float, float]:
    """
    Resolve a location string → (lat, lng).
    Kept as a separate deterministic step rather than letting the LLM guess coordinates —
    LLMs hallucinate plausible-looking but wrong lat/lng values silently.
    """
    logger.info(request_id, "geocode_start", location_name=location_name)
    t0 = time.time()

    resp = requests.get(
        cfg.places.geocode_url,
        params={"address": location_name, "key": api_key},
        timeout=10,
    )
    resp.raise_for_status()

    data = resp.json()
    if data["status"] != "OK" or not data["results"]:
        raise ValueError(f"Geocoding failed for '{location_name}': {data['status']}")

    loc = data["results"][0]["geometry"]["location"]
    lat, lng = loc["lat"], loc["lng"]
    latency_ms = round((time.time() - t0) * 1000, 1)

    logger.info(request_id, "geocode_done", lat=lat, lng=lng, latency_ms=latency_ms)
    return lat, lng


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10), reraise=True)
def nearby_search(params: NearbySearchParams, api_key: str, request_id: str = "") -> List[Place]:
    """
    Recall phase: fetch candidates by proximity using the Places Nearby Search API.
    Location-centric recall means results are bounded by real geography, not keyword
    coincidence — a critical property for 'near me' style queries.
    """
    body: dict = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": params.latitude, "longitude": params.longitude},
                "radius": float(params.radius_meters),
            }
        },
        "maxResultCount": params.max_results,
    }

    if params.included_types:
        body["includedTypes"] = params.included_types

    logger.info(
        request_id, "nearby_search_start",
        location=params.location_name,
        lat=params.latitude, lng=params.longitude,
        radius_m=params.radius_meters,
        types=params.included_types,
    )
    t0 = time.time()

    resp = requests.post(
        cfg.places.nearby_search_url,
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": _NEARBY_FIELD_MASK,
            "Content-Type": "application/json",
        },
        json=body,
        timeout=15,
    )
    resp.raise_for_status()

    latency_ms = round((time.time() - t0) * 1000, 1)
    raw = resp.json().get("places", [])

    places = [
        Place(
            name=p.get("displayName", {}).get("text", "Unknown"),
            address=p.get("formattedAddress", ""),
            rating=p.get("rating"),
            user_rating_count=p.get("userRatingCount"),
            price_level=p.get("priceLevel"),
            maps_url=p.get("googleMapsUri"),
            open_now=p.get("currentOpeningHours", {}).get("openNow"),
            primary_type=p.get("primaryType"),
            reviews=[
                r.get("text", {}).get("text", "")
                for r in p.get("reviews", [])
                if r.get("text", {}).get("text")
            ],
        )
        for p in raw
    ]

    logger.info(
        request_id, "nearby_search_done",
        candidate_count=len(places),
        with_reviews=sum(1 for p in places if p.reviews),
        latency_ms=latency_ms,
    )
    return places
