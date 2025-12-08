# 同一画像を3回（VART summaryを出す）
python3 src/run_npu_unet_image.py \
  --snapshot ./SNAPSHOTS/SNAP.unet_nearest_256 \
  --input ../Dataset/SR_IMG/IMG01.jpg \
  --out   ./results_npu/mask_nearest_img01.png \
  --h 256 --w 256 --thr 0.5 --repeat 3 --zero_copy

# ディレクトリ一括（複数ファイルを1プロセスで連続実行）
python3 src/run_npu_unet_image.py \
  --snapshot ./SNAPSHOTS/SNAP.unet_nearest_256 \
  --input_dir ../Dataset/SR_IMG \
  --glob "*.jpg,*.png" \
  --out_dir ./results_npu_dir \
  --h 256 --w 256 --thr 0.5 --zero_copy

