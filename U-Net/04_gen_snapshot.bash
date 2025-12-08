python3 src/gen_snapshot_image.py \
  --onnx  weights/unet_carvana_nearest_256_fixed.onnx \
  --input ../Dataset/Web/car01.png \
  --snap  SNAPSHOTS/SNAP.unet_carvana_nearest_256 \
  --h 256 --w 256 --thr 0.5 \
  --providers CPUExecutionProvider \
  --repeat 4

