"""
Offline evaluation against the golden set.

Loads data/golden/golden_set.json, runs the recall → filter → rank pipeline
for each entry, computes NDCG@5, and compares against the recorded baseline.

Exits 0 if NDCG@5 is within the regression threshold of baseline.
Exits 1 if NDCG@5 regresses, blocking deploy.sh from proceeding.

Usage:
    python -m map_llm.evaluation.eval
    python -m map_llm.evaluation.eval --update-baseline
    python -m map_llm.evaluation.eval --golden path/to/golden_set.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from map_llm.config import cfg
from map_llm.evaluation.metrics import ndcg_at_k
from map_llm.llm.gemini import init_vertex
from map_llm.models import NearbySearchParams
from map_llm.pipeline.ranker import rank_by_similarity
from map_llm.pipeline.recall import geocode, nearby_search

_REPO_ROOT = Path(__file__).parent.parent.parent
GOLDEN_SET_PATH = _REPO_ROOT / "data/golden/golden_set.json"
BASELINE_PATH = _REPO_ROOT / "data/golden/baseline.json"

K = 5
REGRESSION_THRESHOLD = 0.05  # block promotion if mean NDCG@5 drops more than 5%


def _run_pipeline(entry: dict[str, object], api_key: str) -> list[str]:
    """Run recall → filter → rank for one golden set entry.

    Args:
        entry: A golden set record with city, country, selected_types, description.
        api_key: Google Places API key.

    Returns:
        Ranked place names (top K), or empty list if recall returns nothing.
    """
    request_id = f"eval_{entry['id']}"
    location_name = f"{entry['city']}, {entry['country']}"

    lat, lng = geocode(location_name, api_key, request_id)

    params = NearbySearchParams(
        location_name=location_name,
        latitude=lat,
        longitude=lng,
        radius_meters=cfg.recall.default_radius_meters,
        included_types=entry["selected_types"],  # type: ignore[arg-type]
        keyword=None,
        max_results=cfg.places.max_candidates,
    )
    candidates = nearby_search(params, api_key, request_id)
    if not candidates:
        return []

    # Apply rating filter — keep unrated places (may be new/good)
    filtered = [
        p for p in candidates
        if p.rating is None or (1.0 <= p.rating <= 5.0)
    ]
    if not filtered:
        return []

    ranked = rank_by_similarity(
        description=str(entry["description"]),
        places=filtered,
        top_k=K,
        request_id=request_id,
    )
    return [p.name for p, _ in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline NDCG@5 evaluation against golden set")
    parser.add_argument("--golden", default=str(GOLDEN_SET_PATH), help="Path to golden_set.json")
    parser.add_argument("--baseline", default=str(BASELINE_PATH), help="Path to baseline.json")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current eval score as the new baseline instead of comparing",
    )
    args = parser.parse_args()

    api_key = os.getenv("PLACES_API_KEY")
    if not api_key:
        sys.exit("Error: PLACES_API_KEY not set.")

    golden_path = Path(args.golden)
    if not golden_path.exists():
        sys.exit(f"Golden set not found: {golden_path}")

    init_vertex()

    golden: list[dict] = json.loads(golden_path.read_text())
    print(f"Evaluating {len(golden)} examples  variant_id={cfg.serving.variant_id}  k={K}\n")

    scores: list[float] = []
    for entry in golden:
        ranked_names = _run_pipeline(entry, api_key)
        expected = set(entry["expected_names"])  # type: ignore[arg-type]
        score = ndcg_at_k(ranked_names, expected, k=K)
        scores.append(score)

        hit = any(n in expected for n in ranked_names)
        tag = "HIT " if hit else "MISS"
        print(f"  [{tag}]  {entry['id']}  ndcg@{K}={score:.3f}")
        print(f"         returned:  {ranked_names}")
        print(f"         expected:  {sorted(expected)}")

    mean_ndcg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nmean NDCG@{K} = {mean_ndcg:.4f}  ({len(scores)} queries)")

    baseline_path = Path(args.baseline)

    if args.update_baseline:
        record = {"ndcg_at_k": round(mean_ndcg, 6), "k": K, "n": len(scores), "variant_id": cfg.serving.variant_id}
        baseline_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nBaseline updated → {baseline_path}")
        return

    if not baseline_path.exists():
        print(f"\nNo baseline at {baseline_path}.")
        print("Run with --update-baseline after verifying the golden set to record one.")
        return

    baseline = json.loads(baseline_path.read_text())
    baseline_score: float = baseline["ndcg_at_k"]
    drop = baseline_score - mean_ndcg

    print(f"baseline NDCG@{K} = {baseline_score:.4f}  drop = {drop:+.4f}")

    if drop > REGRESSION_THRESHOLD:
        print(
            f"\nFAIL  NDCG@{K} dropped {drop:.2%} — exceeds {REGRESSION_THRESHOLD:.0%} threshold."
            f"\nPromotion blocked. Investigate before deploying."
        )
        sys.exit(1)

    print(f"PASS  regression within threshold ({REGRESSION_THRESHOLD:.0%}).")


if __name__ == "__main__":
    main()
