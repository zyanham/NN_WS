export VAISW_SNAPSHOT_DIRECTORY=./SNAPSHOTS/SNAP.unetpp_isic2018_256_b1
python3 src/gen_snapshot_image.py \
  --onnx weights/unetpp_isic2018_best_256.onnx \
  --img ../Dataset/isic2018/ISIC_0012292.jpg --size 256 --repeat 20


#export VAISW_SNAPSHOT_DIRECTORY=./SNAPSHOTS/SNAP.unetpp_isic2018_256_b4
#python3 src/gen_snapshot_image.py \
#  --onnx weights/unetpp_isic2018_best_256.onnx \
#  --calibdir ../Dataset/isic2018 --size 256 --repeat 20 \
#  --batch 4

export VAISW_SNAPSHOT_DIRECTORY=./SNAPSHOTS/SNAP.unetpp_segpc2021_256_b1
python3 src/gen_snapshot_image.py \
  --onnx weights/unetpp_segpc2021_best_256.onnx \
  --img ../Dataset/segpc2021/101.bmp --size 256 --repeat 20

#export VAISW_SNAPSHOT_DIRECTORY=./SNAPSHOTS/SNAP.unetpp_segpc2021_256_b4
#python3 src/gen_snapshot_image.py \
#  --onnx weights/unetpp_segpc2021_best_256.onnx \
#  --calibdir ../Dataset/segpc2021 --size 256 --repeat 20 \
#  --batch 4
