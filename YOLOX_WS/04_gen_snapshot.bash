#!/usr/bin/env bash
set -euo pipefail

yolox_cfg_list=(
  "x",
  "l",
  "m",
  "s",
  "tiny",
  "nano"
)

npu_ip_list=(
  "VE2802_NPU_IP_O00_A304_M3",
  "ZU7EV_NPU_IP_1102",
  "ZU7EV_NPU_IP_1108"
)

for npu_ip in "${npu_ip_list[@]}"; do
source ${VITIS_AI_REPO}/npu_ip/settings.sh ${npu_ip}
  for yolox_cfg in "${yolox_cfg_list[@]}"; do
    VAISW_SNAPSHOT_DUMPIOS=5 \
    VAISW_SNAPSHOT_DIRECTORY=SNAPSHOTS/SNAP.${npu_ip}_YOLOX${yolox_cfg}_b1 \
    VAISW_RUNOPTIMIZATION_DDRSHAPE=N_C_H_W_c \
    VAISW_QUANTIZATION_NBIMAGES=1 \
    ./run assets/dog.jpg ${yolox_clg}
  done
done

