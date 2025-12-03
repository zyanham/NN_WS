#!/usr/bin/env bash
# YOLOX 用 venv + 依存 + 公式 weights 自動DL
# 使い方: source setup.bash

set -e

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${THIS_DIR}"

VENV_DIR="${THIS_DIR}/venv"
PYTHON_BIN=python3

echo "[INFO] YOLOX_WS setup start"
echo "[INFO] ROOT = ${THIS_DIR}"

#------------------------------
# venv 作成 & 有効化
#------------------------------
if [ ! -d "${VENV_DIR}" ]; then
  echo "[INFO] Create venv at ${VENV_DIR}"
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel

#------------------------------
# torch / torchvision （まだなら）
#------------------------------
if ! python - <<'PY' 2>/dev/null
import torch
PY
then
  echo "[INFO] torch not found in venv, install..."
  pip install torch torchvision
else
  echo "[INFO] torch already installed in venv"
fi

#------------------------------
# YOLOX リポジトリ clone
#------------------------------
ORIG_DIR="${THIS_DIR}/original"
YOLOX_DIR="${ORIG_DIR}/YOLOX"

mkdir -p "${ORIG_DIR}"

if [ ! -d "${YOLOX_DIR}" ]; then
  echo "[INFO] Clone official YOLOX into ${YOLOX_DIR}"
  git clone https://github.com/Megvii-BaseDetection/YOLOX.git "${YOLOX_DIR}"
else
  echo "[INFO] YOLOX repo already exists at ${YOLOX_DIR}"
fi

cd "${YOLOX_DIR}"

#------------------------------
# onnx-simplifier==0.4.10 問題を避けるため、requirements から外す
#------------------------------
if grep -q "^onnx-simplifier==0\.4\.10" requirements.txt; then
  echo "[INFO] Patch requirements.txt (disable onnx-simplifier==0.4.10)"
  sed -i 's/^onnx-simplifier==0\.4\.10/# onnx-simplifier==0.4.10 (disabled)/' requirements.txt
fi

# YOLOX 依存インストール
pip install -r requirements.txt

# YOLOX 本体を editable install（依存は再解決させない）
pip install -e . --no-build-isolation --no-deps

cd "${THIS_DIR}"

#------------------------------
# weights 自動ダウンロード
#------------------------------
download_file() {
  local url="$1"
  local out="$2"

  if [ -f "${out}" ]; then
    echo "[INFO] Already exists: ${out}"
    return 0
  fi

  echo "[INFO] Download: ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "${out}" "${url}"
  else
    echo "[ERROR] neither wget nor curl is available. Please install one of them." >&2
    return 1
  fi
}

download_yolox_weights() {
  local WDIR="${THIS_DIR}/weights"
  mkdir -p "${WDIR}"

  echo "[INFO] Download YOLOX pretrained weights into ${WDIR}"

  # Standard models
  download_file "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"   "${WDIR}/yolox_s.pth"
  download_file "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth"   "${WDIR}/yolox_m.pth"
  download_file "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"   "${WDIR}/yolox_l.pth"
  download_file "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth"   "${WDIR}/yolox_x.pth"

  # Light models
  download_file "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth" "${WDIR}/yolox_nano.pth"
  download_file "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth" "${WDIR}/yolox_tiny.pth"

  # 必要なら Darknet 版もついでに:
  # download_file "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth" "${WDIR}/yolox_darknet.pth"
}

download_yolox_weights

sed -i 's/torch.onnx._export/torch.onnx.export/' original/YOLOX/tools/export_onnx.py
pip install "onnx>=1.14" "onnxscript>=0.1.0" onnxruntime

echo "[INFO] YOLOX_WS setup finished."
echo "[INFO] To activate venv:  source ${VENV_DIR}/bin/activate"

