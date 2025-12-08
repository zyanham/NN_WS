python src/export_unetpp_onnx.py \
  --weights-dirs weights/isic2018_unetpp weights/segpc2021_unetpp \
  --size 256 --opset 17 --project-root .
