#!/usr/bin/env python3
"""
Run U-Net++ ONNX on a single image (host PC) and save an overlay result.
- Triggers VAISW snapshot generation on the *first* run by importing `zebra_import` *before* onnxruntime.
- Works for both binary (C=1, sigmoid) and multi-class (C>1, softmax) checkpoints.

Examples:
  python scripts/run_host_unetpp_image.py \
      --img ./sample.png \
      --onnx results/unetpp_isic2018_best_256.onnx \
      --size 256 --thr 0.5 --repeat 2

Outputs:
  results/host_unetpp_overlay.png
  (optional) --save-mask でマスクPNGも保存
"""
import os
import time
import argparse

# IMPORTANT: import zebra_import BEFORE onnxruntime to trigger VAISW compiler
#import zebra_import  # noqa: F401
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


def run_once(sess: ort.InferenceSession, img_bgr: np.ndarray, size: int, thr: float,
             alpha: float, input_name: str | None, save_mask: str | None) -> np.ndarray:
    im_in = letterbox_bgr(img_bgr, size)
    x = (im_in.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]  # NCHW

    if input_name is None:
        input_name = sess.get_inputs()[0].name
    t0 = time.time()
    y = sess.run(None, {input_name: x})[0]  # [N,C,H,W]
    dt = (time.time() - t0) * 1000.0

    if y.ndim != 4:
        raise RuntimeError(f"Unexpected output shape {y.shape}, expected [N,C,H,W]")
    _, C, H, W = y.shape

    if C == 1:
        # binary: sigmoid → threshold
        prob = 1.0 / (1.0 + np.exp(-y[0, 0]))  # [H,W]
        mask = (prob >= thr).astype(np.uint8) * 255
        if save_mask:
            cv2.imwrite(save_mask, mask)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(im_in, 1.0 - alpha, mask3, alpha, 0)
    else:
        # multi-class: softmax → argmax
        prob = softmax_channel(y[0], axis=0)  # [C,H,W]
        pred = prob.argmax(axis=0).astype(np.uint8)  # [H,W]
        # simple colorize via OpenCV colormap using class index * step
        step = max(1, 255 // max(1, C - 1))
        cm_in = (pred * step).astype(np.uint8)
        color = cv2.applyColorMap(cm_in, cv2.COLORMAP_JET)
        if save_mask:
            cv2.imwrite(save_mask, color)
        out = cv2.addWeighted(im_in, 1.0 - alpha, color, alpha, 0)

    print(f"[INFO] Inference time: {dt:.2f} ms  |  out shape: C={C}, H={H}, W={W}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Input image path (BGR)")
    ap.add_argument("--onnx", default="results/unetpp_isic2018_best_256.onnx")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for binary (C=1)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha [0..1]")
    ap.add_argument("--repeat", type=int, default=2, help="Run multiple times in one process (VAISW summary workaround)")
    ap.add_argument("--input-name", default=None, help="Override ONNX input name (auto if omitted)")
    ap.add_argument("--save-mask", default=None, help="Optional path to save raw mask/colored mask")
    args = ap.parse_args()

    ensure_dir("results")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])  # zebra_import hooks here

    img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {args.img}")

    out = None
    for i in range(max(1, args.repeat)):
        out = run_once(sess, img, args.size, args.thr, args.alpha, args.input_name, args.save_mask)

    out_path = "results/host_unetpp_overlay.png"
    cv2.imwrite(out_path, out)
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()

