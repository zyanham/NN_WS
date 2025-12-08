python3 src/run_host_unet_onnx_image.py \
  --onnx weights/unet_carvana_nearest_256.onnx \
  --input ../Dataset/Web/car01.png \
  --out results/mask_img01.png \
  --h 256 --w 256 --thr 0.5

