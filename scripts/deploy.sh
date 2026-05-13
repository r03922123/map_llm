#!/usr/bin/env bash
# Phased rollout: shadow (0%) → smoke test → canary (10%) → manual gate → full (100%)
# Usage: scripts/deploy.sh [--full]   pass --full to skip the manual gate and go to 100%
set -euo pipefail

PROJECT="composite-theme-247507"
REGION="us-central1"
SERVICE="map-llm"
REPO="us-central1-docker.pkg.dev/${PROJECT}/cloud-run-source-deploy/${SERVICE}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_URL="https://map-llm-1074296379160.us-central1.run.app"

GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
IMAGE="${REPO}:${GIT_SHA}"

full_rollout=false
[[ "${1:-}" == "--full" ]] && full_rollout=true

echo "=== Deploy ${SERVICE} @ ${GIT_SHA} ==="
echo "    image : ${IMAGE}"
echo "    region: ${REGION}"
echo

# ── 0. Offline eval gate ──────────────────────────────────────────────────────
# Runs NDCG@5 against the golden set before any build or traffic change.
# A regression (>5% drop from baseline.json) exits here and blocks the deploy.
echo "[0/5] Running offline eval gate ..."
python3 -m map_llm.evaluation.eval
echo

# ── 1. Build & push tagged image ─────────────────────────────────────────────
echo "[1/5] Building and pushing ${IMAGE} ..."
gcloud builds submit "${REPO_ROOT}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --tag="${IMAGE}" \
  --quiet

# ── 2. Deploy new revision with 0% traffic (shadow) ──────────────────────────
echo "[2/5] Deploying new revision (no traffic) ..."
gcloud run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --no-traffic \
  --quiet

NEW_REVISION="$(
  gcloud run revisions list \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --service="${SERVICE}" \
    --sort-by="~DEPLOYED" \
    --limit=1 \
    --format="value(metadata.name)"
)"
echo "    new revision: ${NEW_REVISION}"

# ── 3. Smoke test against the stable endpoint ─────────────────────────────────
echo "[3/5] Running smoke test against stable endpoint ..."
python3 "${SCRIPT_DIR}/smoke_test.py" "${BASE_URL}"

# ── 4. Canary: route 10% of traffic to new revision ──────────────────────────
echo "[4/5] Routing 10%% traffic to ${NEW_REVISION} (canary) ..."
gcloud run services update-traffic "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --to-revisions="${NEW_REVISION}=10" \
  --quiet

echo
echo "    Canary is live. Monitor errors and latency for a few minutes."
echo "    Cloud Run metrics: https://console.cloud.google.com/run/detail/${REGION}/${SERVICE}/metrics?project=${PROJECT}"
echo

# ── 5. Full rollout or pause for manual gate ─────────────────────────────────
if $full_rollout; then
  echo "[5/5] --full flag set — routing 100%% traffic to ${NEW_REVISION} ..."
  gcloud run services update-traffic "${SERVICE}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --to-revisions="${NEW_REVISION}=100" \
    --quiet
  echo
  echo "=== Rollout complete. Run smoke test to confirm: ==="
  echo "    python3 scripts/smoke_test.py"
else
  echo "[5/5] Paused at canary (10%). When satisfied, promote with:"
  echo "    gcloud run services update-traffic ${SERVICE} \\"
  echo "      --project=${PROJECT} --region=${REGION} \\"
  echo "      --to-revisions=${NEW_REVISION}=100"
  echo
  echo "    To roll back:"
  echo "    gcloud run services update-traffic ${SERVICE} \\"
  echo "      --project=${PROJECT} --region=${REGION} \\"
  echo "      --to-revisions=${NEW_REVISION}=0"
fi
