#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/exps/debug_boundary_mix_vis/config.yaml"
SMOKE_SPLIT_DIR="${ROOT}/exps/debug_boundary_mix_vis/pascal_smoke"
SRC_SPLIT_DIR="${ROOT}/data/splitsall/pascal_u2pl/662"
SRC_VAL_SPLIT="${ROOT}/data/splitsall/pascal_u2pl/val.txt"
DATA_ROOT="${ROOT}/data/VOC2012"
PORT="${PORT:-53928}"

export PATH=/venv/main/bin:$PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled

PYTHON_BIN=${PYTHON_BIN:-/venv/main/bin/python}
TORCHRUN=${TORCHRUN:-/venv/main/bin/torchrun}

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing dataset directory: ${DATA_ROOT}" >&2
  echo "Expected VOC layout includes JPEGImages/ and SegmentationClassAug/." >&2
  exit 2
fi

if [[ ! -f "${SRC_VAL_SPLIT}" && -f "${SRC_SPLIT_DIR}/val.txt" ]]; then
  SRC_VAL_SPLIT="${SRC_SPLIT_DIR}/val.txt"
fi

if [[ ! -f "${SRC_SPLIT_DIR}/labeled.txt" || ! -f "${SRC_SPLIT_DIR}/unlabeled.txt" || ! -f "${SRC_VAL_SPLIT}" ]]; then
  echo "Missing source split files under: ${SRC_SPLIT_DIR}" >&2
  echo "Expected labeled.txt, unlabeled.txt, and parent or local val.txt." >&2
  exit 2
fi

mkdir -p "${SMOKE_SPLIT_DIR}"
sed -n '1p' "${SRC_SPLIT_DIR}/labeled.txt" > "${SMOKE_SPLIT_DIR}/labeled.txt"
sed -n '1p' "${SRC_SPLIT_DIR}/unlabeled.txt" > "${SMOKE_SPLIT_DIR}/unlabeled.txt"
sed -n '1p' "${SRC_VAL_SPLIT}" > "${SMOKE_SPLIT_DIR}/val.txt"

first_id="$(sed -n '1p' "${SMOKE_SPLIT_DIR}/labeled.txt")"
if [[ -z "${first_id}" ]]; then
  echo "Smoke labeled split is empty: ${SMOKE_SPLIT_DIR}/labeled.txt" >&2
  exit 2
fi

if [[ ! -f "${DATA_ROOT}/JPEGImages/${first_id}.jpg" ]]; then
  echo "Missing smoke image: ${DATA_ROOT}/JPEGImages/${first_id}.jpg" >&2
  exit 2
fi

if [[ ! -f "${DATA_ROOT}/SegmentationClassAug/${first_id}.png" ]]; then
  echo "Missing smoke mask: ${DATA_ROOT}/SegmentationClassAug/${first_id}.png" >&2
  exit 2
fi

cd "${ROOT}"
"${TORCHRUN}" --standalone --nproc_per_node=1 --master_port="${PORT}" \
  "${ROOT}/train_semi.py" \
  --config="${CONFIG}" \
  --seed 2 \
  --port "${PORT}"
