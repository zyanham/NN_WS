#!/usr/bin/env bash
set -euo pipefail

NN_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${NN_ROOT}"

source "${NN_ROOT}/setup.bash"

MODEL="${1:-nano}"
TSIZE="${2:-416}"
CAMID="${3:-0}"  # カメラデバイスID

YOLOX_ROOT="${NN_ROOT}/original/YOLOX"
CKPT="${NN_ROOT}/weights/yolox_${MODEL}.pth"

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

python "${YOLOX_ROOT}/tools/demo.py" webcam \
  -f "${YOLOX_ROOT}/exps/default/yolox_${MODEL}.py" \
  -c "${CKPT}" \
  --conf 0.3 \
  --nms 0.45 \
  --tsize "${TSIZE}" \
  --device cpu \
  --camid "${CAMID}"

