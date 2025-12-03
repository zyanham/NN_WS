#!/usr/bin/env bash
set -euo pipefail

# YOLOX_WS のルート
NN_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${NN_ROOT}"

# 事前に 02_run_host_image.bash で使っている setup.bash を再利用
# これで venv 有効化 & YOLOX インストールなどが揃う前提

# 引数
MODEL="${1:-nano}"        # nano / tiny / s / m / l / x
IMGSZ="${2:-416}"         # 416 / 640 など
IMG_ARG="${3:-../../NN_WS_v5.1/Dataset/coco2017/Dataset/COCO30}"  # キャリブ用画像ディレクトリ

# SNAPSHOT 出力先ルート
SNAP_ROOT="${NN_ROOT}/SNAPSHOTS"
mkdir -p "${SNAP_ROOT}"

# ★スナップショット名（NPU IP 名はとりあえず手動運用にしておく）
SNAP_NAME="SNAP.V5.1_YOLOX_${MODEL}_${IMGSZ}"

export VAISW_SNAPSHOT_DIRECTORY="${SNAP_ROOT}/${SNAP_NAME}"
# サマリを出したければコメント解除
# export VAISW_RUNSESSION_SUMMARY=1

echo "[INFO] YOLOX_WS snapshot generation"
echo "[INFO] ROOT                    : ${NN_ROOT}"
echo "[INFO] MODEL                   : ${MODEL}"
echo "[INFO] INPUT SIZE              : ${IMGSZ}"
echo "[INFO] IMAGE / DIR FOR CALIB   : ${IMG_ARG}"
echo "[INFO] VAISW_SNAPSHOT_DIRECTORY: ${VAISW_SNAPSHOT_DIRECTORY}"
echo

# 注意:
#  - ここで VAISW の NPU スタックはすでに有効になっている想定
#    → 事前に 'source npu_ip/settings.sh <IP名>' を実行しておくこと
#  - src/run_host_image.py は PyTorch (PTH) 実装版を使う

python src/run_host_image.py \
  --model "${MODEL}" \
  --imgsz "${IMGSZ}" \
  --image "${IMG_ARG}" \
  --score 0.1

echo
echo "[INFO] Snapshot run finished."
echo "[INFO] Snapshots should be under:"
echo "       ${VAISW_SNAPSHOT_DIRECTORY}"

