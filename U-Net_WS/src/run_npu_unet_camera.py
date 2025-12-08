#!/usr/bin/env python3
# scripts/run_npu_unet_camera.py
import argparse, os, time
import numpy as np
import cv2
import VART

def _quantize_to_nhwc(model, x_nchw_f32_list):
    out = []
    for arr, coeff, dtype, fmt in zip(
        x_nchw_f32_list, model.input_coeffs, model.input_native_types, model.input_shape_formats
    ):
        q = (arr * coeff).round().clip(-128, 127).astype(dtype)
        if fmt == "NCHW":
            q = np.transpose(q, (0, 2, 3, 1))
        out.append(q)
    return out

def _copy_to_native(native_bufs, q_list):
    I = len(q_list)
    for i in range(I):
        N = q_list[i].shape[0]
        for n in range(N):
            np.copyto(native_bufs[n * I + i], q_list[i][n])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_dir", default="results_npu_cam")
    ap.add_argument("--backend", choices=["v4l2","auto"], default="v4l2",
                    help="カメラバックエンド。既定はV4L2でGStreamer回避")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model = VART.Runner(snapshot_dir=args.snapshot, aie_only=False)

    # enable zero-copy input once
    if model.set_input_zero_copy(True):
        raise AttributeError("入力のゼロコピーを有効化できませんでした（CPUリンクあり）。")
    native_bufs = model.alloc_ddr_bufs()

    # Open camera (prefer V4L2)
    if args.backend == "v4l2":
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError("camera open failed")

    # try to set resolution (best effort)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)

    # warmup 1
    ok, frm = cap.read()
    if not ok:
        raise RuntimeError("camera read failed at warmup")
    frm_rs = cv2.resize(frm, (args.w, args.h), interpolation=cv2.INTER_AREA)
    x = (cv2.cvtColor(frm_rs, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0).transpose(2,0,1)[None,...]
    q_list = _quantize_to_nhwc(model, [x])
    _copy_to_native(native_bufs, q_list)
    _ = model(native_bufs)

    # loop
    t0 = time.time(); n = 0
    while n < args.frames:
        ok, frm = cap.read()
        if not ok:
            break
        frm_rs = cv2.resize(frm, (args.w, args.h), interpolation=cv2.INTER_AREA)
        x = (cv2.cvtColor(frm_rs, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0).transpose(2,0,1)[None,...]
        q_list = _quantize_to_nhwc(model, [x])
        _copy_to_native(native_bufs, q_list)
        out_native = model(native_bufs)

        # dequantize -> FP32(NCHW)
        arr = np.asarray(out_native[0])
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(model.output_coeffs[0])
        fmt = model.output_shape_formats[0] if model.output_shape_formats else "NCHW"
        if arr.ndim == 4 and fmt == "NHWC":
            arr = np.transpose(arr, (0, 3, 1, 2))
        y = 1.0/(1.0+np.exp(-arr)); y = y[0,0]
        mask = (y >= args.thr).astype(np.uint8)*255
        cv2.imwrite(os.path.join(args.save_dir, f"mask_{n:05d}.png"), mask)
        n += 1

    cap.release()
    dt = time.time() - t0
    print(f"[INFO] {n} frames in {dt:.2f}s -> { (n/max(dt,1e-6)):.2f} FPS")

if __name__ == "__main__":
    main()

