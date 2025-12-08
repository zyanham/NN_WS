#!/usr/bin/env python3
"""
U-Net++ snapshot runner on VEK280 using VART directly (USB camera).
This calls the snapshot via VART.Runner (NPU). Display can be disabled with --no-display.

Example:
  source /etc/vai.sh
  python3 scripts/run_vart_unetpp_camera.py \
      --snapshot SNAP.unetpp_isic2018 --device 0 --size 256

Keys (when display enabled):
  q : quit
  s : save current overlay to results/vart_cam_*.png
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import VART


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def letterbox_bgr(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    pad_t = (size - nh) // 2
    pad_b = size - nh - pad_t
    pad_l = (size - nw) // 2
    pad_r = size - nw - pad_l
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img_p = cv2.copyMakeBorder(img_r, pad_t, pad_b, pad_l, pad_r,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img_p


def softmax_channel(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def colorize(pred: np.ndarray, C: int) -> np.ndarray:
    step = max(1, 255 // max(1, C - 1))
    cm_in = (pred * step).astype(np.uint8)
    return cv2.applyColorMap(cm_in, cv2.COLORMAP_JET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    ensure_dir("results")

    model = VART.Runner(snapshot_dir=args.snapshot, aie_only=False)
    in_fmt = model.input_shape_formats[0]
    out_fmt = model.output_shape_formats[0]

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {args.device}")

    ema_fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        base = letterbox_bgr(frame, args.size)

        x = (base.astype(np.float32) / 255.0)
        if in_fmt == "NCHW":
            x = x.transpose(2, 0, 1)[None, ...]
        elif in_fmt == "NHWC":
            x = x[None, ...]
        else:
            raise RuntimeError(f"Unsupported input format: {in_fmt}")

        t0 = time.time()
        out_list = model([x])
        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        ema_fps = fps if ema_fps == 0 else 0.9 * ema_fps + 0.1 * fps

        y = out_list[0]
        if out_fmt == "NHWC":
            y = y.transpose(0, 3, 1, 2)
        _, C, H, W = y.shape

        if C == 1:
            prob = 1.0 / (1.0 + np.exp(-y[0, 0]))
            mask = (prob >= args.thr).astype(np.uint8) * 255
            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(base, 1.0 - args.alpha, mask3, args.alpha, 0)
        else:
            prob = softmax_channel(y[0], axis=0)
            pred = prob.argmax(axis=0).astype(np.uint8)
            color = colorize(pred, C)
            overlay = cv2.addWeighted(base, 1.0 - args.alpha, color, args.alpha, 0)

        if args.no-display:
            ts = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join('results', f'vart_cam_{ts}.png')
            cv2.imwrite(path, overlay)
            print(f"[FPS {ema_fps:.1f}] saved: {path}")
        else:
            cv2.putText(overlay, f"FPS: {ema_fps:.1f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("U-Net++ (NPU/VART)", overlay)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                ts = time.strftime('%Y%m%d_%H%M%S')
                path = os.path.join('results', f'vart_cam_{ts}.png')
                cv2.imwrite(path, overlay)
                print(f"[OK] saved: {path}")

    cap.release()
    if not args.no-display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

