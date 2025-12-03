#!/usr/bin/env bash
set -euo pipefail

NN_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${NN_ROOT}"

# venv 有効化（setup.bash は再実行しても大丈夫な作り）
source "${NN_ROOT}/setup.bash"

MODEL="${1:-nano}"   # nano/tiny/s/m/l/x
TSIZE="${2:-416}"    # 416 / 640 とか
IMG_PATH="${3:-../../NN_WS_v5.1/Dataset/coco2017/Dataset/COCO30}"  # 1枚 or ディレクトリ

YOLOX_ROOT="${NN_ROOT}/original/YOLOX"
CKPT="${NN_ROOT}/weights/yolox_${MODEL}.pth"

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

python "${YOLOX_ROOT}/tools/demo.py" image \
  -f "${YOLOX_ROOT}/exps/default/yolox_${MODEL}.py" \
  -c "${CKPT}" \
  --path "${IMG_PATH}" \
  --conf 0.3 \
  --nms 0.45 \
  --tsize "${TSIZE}" \
  --device cpu \
  --save_result

