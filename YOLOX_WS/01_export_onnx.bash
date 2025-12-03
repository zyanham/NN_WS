#!/usr/bin/env bash
# YOLOX -> ONNX エクスポート用
# 使い方:
#   ./01_export_onnx.bash nano 640
#   ./01_export_onnx.bash s    640  など

set -e

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${THIS_DIR}"

# venv 有効化
# shellcheck disable=SC1091
source "${THIS_DIR}/venv/bin/activate"

SIZE="${1:-nano}"   # nano / tiny / s / m / l / x
IMGSZ="${2:-640}"   # 入力解像度

python src/export_yolox_onnx.py \
  --size "${SIZE}" \
  --img-size "${IMGSZ}"

