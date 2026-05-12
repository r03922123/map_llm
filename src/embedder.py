import math
import time

from google import genai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config import cfg
from .logger import StructuredLogger
from .models import Place

logger = StructuredLogger("embedder")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=cfg.gcp.project, location=cfg.gcp.location)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10), reraise=True)
def embed_texts(texts: list[str], request_id: str = "") -> list[list[float]]:
    """
    Batch embed all texts in a single API call.
    Batching is critical: N sequential calls would be N× slower and cost N× more.
    text-embedding-004 charges per character, not per call — batching has no cost penalty.
    """
    logger.info(request_id, "embed_start", count=len(texts), model=cfg.embedding.model)
    t0 = time.time()

    response = _get_client().models.embed_content(
        model=cfg.embedding.model,
        contents=texts,
    )

    latency_ms = round((time.time() - t0) * 1000, 1)
    vectors = [e.values for e in response.embeddings]
    logger.info(
        request_id, "embed_done",
        count=len(vectors),
        dims=len(vectors[0]) if vectors else 0,
        latency_ms=latency_ms,
    )
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_place_text(p: Place) -> str:
    """
    Concatenate all available text signals for a place into one string for embedding.
    Order: type → name → address → reviews (most semantic signal last, recency-weighted).
    Fallback when reviews are empty: name + type + address still gives useful signal
    for geo/category similarity, just not vibe/atmosphere similarity.
    """
    parts = [p.primary_type or "", p.name, p.address]
    parts.extend(p.reviews)  # up to 5 review snippets
    return ". ".join(filter(None, parts))


def rank_by_similarity(
    description: str,
    places: list[Place],
    top_k: int,
    request_id: str,
) -> list[tuple[Place, float]]:
    """
    Embed user description + all place texts in one batch call, then rank by cosine similarity.
    Returns (place, score) pairs sorted descending, capped at top_k.
    """
    place_texts = [build_place_text(p) for p in places]
    all_texts = [description] + place_texts

    vectors = embed_texts(all_texts, request_id)
    user_vec = vectors[0]
    place_vecs = vectors[1:]

    scored = sorted(
        ((places[i], cosine_similarity(user_vec, place_vecs[i])) for i in range(len(places))),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info(
        request_id, "similarity_ranked",
        top_scores=[round(s, 3) for _, s in scored[:top_k]],
    )
    return scored[:top_k]
