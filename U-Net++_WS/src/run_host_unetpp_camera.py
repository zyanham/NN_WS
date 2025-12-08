#!/usr/bin/env python3
"""
Run U-Net++ ONNX on a USB camera (host PC) with live overlay.
- Triggers VAISW snapshot generation on the first frames by importing `zebra_import` before onnxruntime.
- Handles both binary (C=1) and multi-class (C>1) heads.

Examples:
  python scripts/run_host_unetpp_camera.py \
      --onnx results/unetpp_isic2018_best_256.onnx \
      --size 256 --thr 0.5 --device 0

Keys:
  q : quit
  s : save current frame to results/cam_YYYYmmdd_HHMMSS.png
"""
import os
import time
import argparse

# IMPORTANT: import zebra_import BEFORE onnxruntime
# import zebra_import  # noqa: F401
import onnxruntime as ort

import cv2
import numpy as np


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


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


def colorize_from_pred(pred: np.ndarray, C: int) -> np.ndarray:
    # pred: [H,W] int
    step = max(1, 255 // max(1, C - 1))
    cm_in = (pred * step).astype(np.uint8)
    return cv2.applyColorMap(cm_in, cv2.COLORMAP_JET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="weights/unetpp_isic2018_best_256.onnx")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for binary (C=1)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha [0..1]")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0.0, help="Try to set camera FPS if >0")
    ap.add_argument("--width", type=int, default=0, help="Try to set camera width")
    ap.add_argument("--height", type=int, default=0, help="Try to set camera height")
    ap.add_argument("--input-name", default=None)
    args = ap.parse_args()

    ensure_dir("results")

    cap = cv2.VideoCapture(args.device)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {args.device}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])  # zebra_import hooks
    input_name = args.input_name or sess.get_inputs()[0].name

    # Prime one frame to detect class count
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from camera")
    im_in = letterbox_bgr(frame, args.size)
    x = (im_in.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    y = sess.run(None, {input_name: x})[0]
    if y.ndim != 4:
        raise RuntimeError(f"Unexpected output shape {y.shape}")
    _, C, H, W = y.shape

    prev = time.time()
    ema_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()

        im_in = letterbox_bgr(frame, args.size)
        x = (im_in.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        y = sess.run(None, {input_name: x})[0]

        if C == 1:
            prob = 1.0 / (1.0 + np.exp(-y[0, 0]))
            mask = (prob >= args.thr).astype(np.uint8) * 255
            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(im_in, 1.0 - args.alpha, mask3, args.alpha, 0)
        else:
            prob = softmax_channel(y[0], axis=0)
            pred = prob.argmax(axis=0).astype(np.uint8)
            color = colorize_from_pred(pred, C)
            out = cv2.addWeighted(im_in, 1.0 - args.alpha, color, args.alpha, 0)

        dt = time.time() - t0
        fps = 1.0 / max(1e-6, dt)
        ema_fps = 0.9 * ema_fps + 0.1 * fps if ema_fps > 0 else fps

        cv2.putText(out, f"FPS: {ema_fps:.1f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("U-Net++ (host)", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            ts = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join('results', f'cam_{ts}.png')
            cv2.imwrite(path, out)
            print(f"[OK] saved: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

