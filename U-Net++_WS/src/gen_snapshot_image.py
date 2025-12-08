#!/usr/bin/env python3
"""
Generate a VAISW snapshot (Vitis AI 2025.1) by running ONNX inference on images.
- IMPORTANT: `zebra_import` must be imported BEFORE `onnxruntime` to trigger compilation.
- NEW:
  * `--batch` でバッチサイズを指定可能（N>1で複数バッチ検証が可能）。
  * `--calibdir` 指定時は **デフォルトでディレクトリ内の全画像**を使用（`--num` が 0 のとき）。
  * `--recursive` と `--exts` で探索方法を調整可能。

Usage examples:
  # 単一画像、2回（1プロセス）まわしてSNAP作成
  python scripts/gen_snapshot_image.py \
      --onnx results/unetpp_isic2018_best_256.onnx \
      --img ./sample.png --size 256 --repeat 2

  # ディレクトリ内の全画像で、バッチ=4 で実行（繰り返し1回）
  python scripts/gen_snapshot_image.py \
      --onnx results/unetpp_isic2018_best_256.onnx \
      --calibdir ./Dataset/CalibImages --size 256 --batch 4 --repeat 1

  # サブディレクトリも含めて探索、拡張子を限定
  python scripts/gen_snapshot_image.py \
      --onnx results/unetpp_isic2018_best_256.onnx \
      --calibdir ./Dataset/CalibImages --recursive --exts .jpg,.png \
      --size 256 --batch 8 --repeat 1

Notes:
- Make sure to set VAISW_SNAPSHOT_DIRECTORY prior to running, e.g.:
    export VAISW_SNAPSHOT_DIRECTORY=$PWD/SNAP
- You should have sourced VAISW before running:
    source $VAI_SW/settings.sh
- VAISW の "Accumulation" が 4 サンプル必要な場合は、
  `--batch 4 --repeat 1` や `--batch 2 --repeat 2` のように合計を満たしてください。
"""
import os
import sys
import time
from pathlib import Path
import argparse
import glob

# Trigger VAISW compiler hooks
import zebra_import  # noqa: F401
import onnxruntime as ort

import cv2
import numpy as np


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


def collect_images(calibdir: str | None, img: str | None, recursive: bool, exts: list[str], num: int) -> list[str]:
    paths: list[str] = []
    if calibdir:
        root = Path(calibdir)
        if recursive:
            it = root.rglob('*')
        else:
            it = root.glob('*')
        exset = set(e.lower() for e in exts)
        for p in it:
            if p.is_file() and p.suffix.lower() in exset:
                paths.append(str(p))
        paths.sort()
        if num and num > 0:
            paths = paths[:num]
    elif img:
        paths = [img]
    else:
        raise ValueError("Specify --calibdir or --img")
    if not paths:
        raise FileNotFoundError("No input images found")
    return paths


def chunk_list(lst: list[str], chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def list_recent_snapshots(root: Path, topk: int = 5) -> list[Path]:
    if not root.exists():
        return []
    cands = []
    for p in root.glob("**/"):
        try:
            mtime = p.stat().st_mtime
            cands.append((mtime, p))
        except Exception:
            pass
    cands.sort(reverse=True)
    return [p for _, p in cands[:topk]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1, help="Batch size per inference call")
    ap.add_argument("--repeat", type=int, default=2, help="Repeat per batch in same process")
    ap.add_argument("--input-name", default=None, help="Override ONNX input name")
    ap.add_argument("--calibdir", default=None, help="Directory of images for calibration")
    ap.add_argument("--img", default=None, help="Single image path (alternative to --calibdir)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories under --calibdir")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.tif,.tiff", help="Comma-separated extensions to include")
    ap.add_argument("--num", type=int, default=0, help="Use only first N images (0=use all)")
    args = ap.parse_args()

    snapshot_root = os.environ.get("VAISW_SNAPSHOT_DIRECTORY", None)
    if not snapshot_root:
        print("[WARN] VAISW_SNAPSHOT_DIRECTORY is not set; snapshots may go to the default VAISW path.")
    else:
        print(f"[INFO] VAISW_SNAPSHOT_DIRECTORY = {snapshot_root}")

    exts = [e.strip() for e in args.exts.split(',') if e.strip()]
    paths = collect_images(args.calibdir, args.img, args.recursive, exts, args.num)
    print(f"[INFO] Using {len(paths)} image(s) for snapshot generation")

    # Prepare session (zebra_import already active)
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    input_name = args.input_name or sess.get_inputs()[0].name

    total_batches = 0
    total_images = 0
    t_total_ms = 0.0

    def prep_one(pth: str) -> np.ndarray:
        img = cv2.imread(pth, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(pth)
        im_in = letterbox_bgr(img, args.size)
        x = (im_in.astype(np.float32) / 255.0).transpose(2, 0, 1)  # 3xHxW
        return x

    for bi, group in enumerate(chunk_list(paths, max(1, args.batch)), 1):
        arrs = []
        for p in group:
            try:
                arrs.append(prep_one(p))
            except Exception as e:
                print(f"[WARN] Skipping unreadable: {p} ({e})")
        if not arrs:
            continue
        x = np.stack(arrs, axis=0).astype(np.float32)  # Bx3xHxW

        for r in range(max(1, args.repeat)):
            t0 = time.time()
            _ = sess.run(None, {input_name: x})
            dt_ms = (time.time() - t0) * 1000.0
            t_total_ms += dt_ms
            total_batches += 1
        total_images += len(arrs)

        if (bi % 5 == 0) or (bi == (len(paths) - 1) // max(1, args.batch) + 1):
            avg_per_batch = t_total_ms / max(1, total_batches)
            avg_per_img = (t_total_ms / max(1, total_batches)) / max(1, len(arrs))
            print(f"[INFO] batch {bi} | last {dt_ms:.2f} ms | avg/batch {avg_per_batch:.2f} ms | est avg/img {avg_per_img:.2f} ms")

    if total_batches > 0:
        print(f"[OK] Finished {total_batches} batch runs over {total_images} image(s); avg per-batch {t_total_ms/total_batches:.2f} ms")

    # Report likely snapshot locations
    if snapshot_root:
        root = Path(snapshot_root)
        recent = list_recent_snapshots(root)
        if recent:
            print("[INFO] Recent snapshot-related directories (most recent first):")
            for p in recent:
                try:
                    contains = []
                    if (p / "snapshot.dump").exists():
                        contains.append("snapshot.dump")
                    if (p / "wrp_network_iriz.onnx").exists():
                        contains.append("wrp_network_iriz.onnx")
                    if (p / "wrp_network").exists():
                        contains.append("wrp_network/")
                    print(f"  - {p}  [{', '.join(contains) if contains else '...'}]")
                except Exception:
                    print(f"  - {p}")
        else:
            print(f"[INFO] No directories found yet under {snapshot_root}. If this is the first run, try increasing --batch or --repeat.")


if __name__ == "__main__":
    main()

