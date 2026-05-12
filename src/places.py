import time
from typing import List

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .logger import StructuredLogger
from .models import Place

logger = StructuredLogger("places")

_ENDPOINT = "https://places.googleapis.com/v1/places:searchText"
_FIELD_MASK = ",".join([
    "places.displayName",
    "places.formattedAddress",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.googleMapsUri",
    "places.currentOpeningHours",
])


@retry(
    stop=stop_after_attempt(3),
    # Full jitter exponential backoff — de-correlates retries under load (AWS Marc Brooker, 2015)
    wait=wait_random_exponential(multiplier=1, max=10),
    reraise=True,
)
def search_places(
    query: str,
    api_key: str,
    max_results: int = 20,
    request_id: str = "",
) -> List[Place]:
    logger.info(request_id, "places_search_start", query=query, max_results=max_results)
    t0 = time.time()

    resp = requests.post(
        _ENDPOINT,
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": _FIELD_MASK,
            "Content-Type": "application/json",
        },
        json={"textQuery": query, "maxResultCount": max_results},
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
        )
        for p in raw
    ]

    logger.info(
        request_id,
        "places_search_done",
        candidate_count=len(places),
        latency_ms=latency_ms,
    )
    return places
