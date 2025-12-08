python3 src/run_npu_unet_camera.py \
  --snapshot ./SNAPSHOTS/SNAP.unet_nearest_256 \
  --device 0 --frames 200 \
  --h 256 --w 256 --thr 0.5 \
  --backend v4l2

