python3 src/run_host_unetpp_image.py \
  --img ../Dataset/isic2018/ISIC_0012169.jpg \
  --onnx weights/unetpp_isic2018_best_256.onnx \
  --size 256 --thr 0.5 --repeat 4

python3 src/run_host_unetpp_image.py \
  --img ../Dataset/segpc2021/101.bmp \
  --onnx weights/unetpp_segpc2021_best_256.onnx \
  --size 256 --thr 0.5 --repeat 4
