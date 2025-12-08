python3 src/export_unet_carvana_nearest_onnx.py \
	--out weights/unet_carvana_nearest_256.onnx \
	--h 256 --w 256 --scale 0.5

# ONNX内の全Resizeを NPU向けに統一
python3 src/patch_resize_attrs.py \
	--in  weights/unet_carvana_nearest_256.onnx \
	--out weights/unet_carvana_nearest_256_fixed.onnx
