#!/usr/bin/env python3
# scripts/run_npu_unet_image.py
import argparse, os, time, glob
import numpy as np
import cv2
import VART

def _quantize_to_nhwc(model, x_nchw_f32_list):
    """FP32(NCHW,0..1) -> int*(NHWC) per input tensor (usually 1 tensor)"""
    out = []
    for arr, coeff, dtype, fmt in zip(
        x_nchw_f32_list, model.input_coeffs, model.input_native_types, model.input_shape_formats
    ):
        # quantize
        q = (arr * coeff).round().clip(-128, 127).astype(dtype)
        # NCHW->NHWC if needed
        if fmt == "NCHW":
            q = np.transpose(q, (0, 2, 3, 1))
        out.append(q)
    return out

def _run_one(model, img_path, size_hw, thr, out_path, native_bufs=None):
    # load & preprocess
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = size_hw
    if h > 0 and w > 0:
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,H,W)

    # quantize -> (NHWC,int*)
    q_list = _quantize_to_nhwc(model, [x])

    # prefer zero-copy buffers if provided
    if native_bufs is not None:
        I = len(q_list)
        for i in range(I):
            N = q_list[i].shape[0]
            for n in range(N):
                np.copyto(native_bufs[n * I + i], q_list[i][n])
        inp = native_bufs
    else:
        inp = q_list

    t0 = time.time()
    out_native = model(inp)
    dt = (time.time() - t0) * 1000.0

    # dequantize -> FP32(NCHW)
    outs = []
    for i, out in enumerate(out_native):
        arr = np.asarray(out)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(model.output_coeffs[i])
        fmt = model.output_shape_formats[i] if i < len(model.output_shape_formats) else "NCHW"
        if arr.ndim == 4 and fmt == "NHWC":
            arr = np.transpose(arr, (0, 3, 1, 2))
        outs.append(arr)

    y = 1.0 / (1.0 + np.exp(-outs[0]))  # logits -> prob
    y = y[0, 0]
    mask = (y >= thr).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mask)
    return dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="SNAP.* ディレクトリ")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="単一画像パス")
    g.add_argument("--input_dir", help="ディレクトリ内画像を一括処理")
    ap.add_argument("--glob", default="*.jpg,*.jpeg,*.png,*.bmp",
                    help="--input_dir利用時の拡張子（カンマ区切り）")
    ap.add_argument("--out", default="results_npu/mask.png",
                    help="--input時の保存先、--input_dir時は out_dir を使う")
    ap.add_argument("--out_dir", default="results_npu_dir",
                    help="--input_dir時の保存ディレクトリ")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--repeat", type=int, default=1,
                    help="同一画像の繰り返し回数（VART summary用に2以上推奨）")
    ap.add_argument("--zero_copy", action="store_true",
                    help="DDRゼロコピー（ネイティブ入力バッファ再利用）を有効化")
    args = ap.parse_args()

    model = VART.Runner(snapshot_dir=args.snapshot, aie_only=False)

    native_bufs = None
    if args.zero_copy:
        if model.set_input_zero_copy(True):
            raise AttributeError("入力のゼロコピーを有効化できませんでした（CPUリンクあり）。")
        native_bufs = model.alloc_ddr_bufs()

        # 出力もFP32復元せずネイティブのまま読みたければ以下も
        # if model.set_output_zero_copy(True):
        #     raise AttributeError("出力のゼロコピーを有効化できませんでした。")

    size = (args.h, args.w)
    times = []

    if args.input:
        for i in range(max(1, args.repeat)):
            dt = _run_one(model, args.input, size, args.thr, args.out, native_bufs)
            times.append(dt)
            print(f"[INFO] NPU inference #{i+1}: {dt:.2f} ms -> {args.out}")
    else:
        patterns = [p.strip() for p in args.glob.split(",")]
        img_list = []
        for p in patterns:
            img_list += glob.glob(os.path.join(args.input_dir, p))
        img_list = sorted(list(dict.fromkeys(img_list)))  # unique & sorted
        if not img_list:
            raise FileNotFoundError(f"No images under {args.input_dir} for {args.glob}")
        os.makedirs(args.out_dir, exist_ok=True)
        run_idx = 0
        for img_path in img_list:
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.out_dir, f"{base}_mask.png")
            dt = _run_one(model, img_path, size, args.thr, out_path, native_bufs)
            times.append(dt)
            run_idx += 1
            print(f"[INFO] [{run_idx}/{len(img_list)}] {os.path.basename(img_path)} -> {dt:.2f} ms")

    if len(times) >= 2:
        print(f"[STAT] runs={len(times)}  avg={np.mean(times):.2f} ms  "
              f"min={np.min(times):.2f} ms  max={np.max(times):.2f} ms")
    else:
        print("[STAT] 1回のみ（VART詳細サマリは2回以上で表示されます）")

if __name__ == "__main__":
    main()

