#!/usr/bin/env python3
"""
Batch U-Net++ snapshot runner on VEK280 using VART (process all images in a directory).
- VART.Runner を用いて NPU 実行（スナップショット必須）
- 入力ディレクトリ配下の画像を一括推論し、results 以下に結果PNGを書き戻し
- --batch で実行バッチ数を指定（0=スナップショットのバッチを自動使用）

Usage examples:
  source /etc/vai.sh
  python3 scripts/run_npu_unetpp_image.py \
      --snapshot SNAP.unetpp_isic2018_256_b4 \
      --indir Dataset/isic2018 \
      --size 256 --thr 0.5 --outdir results --suffix _overlay --batch 4 --recursive --mirror

Options:
  --snapshot   : SNAPディレクトリへのパス（例: SNAP.unetpp_isic2018_256_b4）
  --indir      : 入力画像ディレクトリ
  --size       : ネット入力サイズ（正方）
  --thr        : C=1 のときの閾値
  --alpha      : オーバーレイの強さ（0..1）
  --outdir     : 出力先のトップ（既定: results）
  --suffix     : 出力ファイル名に付けるサフィックス（既定: _overlay）
  --exts       : 対象拡張子カンマ区切り（例: .jpg,.png,.bmp; 既定: 汎用セット）
  --recursive  : 再帰的に探索（付けなければ直下のみ）
  --mirror     : 入力の相対ディレクトリ構造を outdir 配下にミラー保存
  --save-mask  : オーバーレイに加えて、マスク画像も保存
  --batch      : 実行時のバッチサイズ（0=スナップショットのバッチを自動使用）
"""
import argparse
from pathlib import Path
import time
import os
import sys

import cv2
import numpy as np
import VART


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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


def overlay_from_logits(logits_1n: np.ndarray, fmt: str, thr: float, alpha: float, base_bgr: np.ndarray):
    """
    logits_1n: [1,C,H,W] or [1,H,W,C]
    return: (overlay_bgr, mask_like_bgr_or_gray)
    """
    if fmt == "NHWC":
        logits_1n = logits_1n.transpose(0, 3, 1, 2)
    _, C, H, W = logits_1n.shape
    if C == 1:
        prob = 1.0 / (1.0 + np.exp(-logits_1n[0, 0]))
        mask = (prob >= thr).astype(np.uint8) * 255
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base_bgr, 1.0 - alpha, mask3, alpha, 0)
        return out, mask
    else:
        prob = softmax_channel(logits_1n[0], axis=0)
        pred = prob.argmax(axis=0).astype(np.uint8)
        step = max(1, 255 // max(1, C - 1))
        cm_in = (pred * step).astype(np.uint8)
        color = cv2.applyColorMap(cm_in, cv2.COLORMAP_JET)
        out = cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0)
        return out, color


def gather_images(indir: Path, recursive: bool, exts_set: set[str]):
    if recursive:
        files = [p for p in indir.rglob('*') if p.suffix.lower() in exts_set]
    else:
        files = [p for p in indir.glob('*') if p.suffix.lower() in exts_set]
    files.sort()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--indir', required=True)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--suffix', default='_overlay')
    ap.add_argument('--exts', default='.jpg,.jpeg,.png,.bmp,.tif,.tiff')
    ap.add_argument('--recursive', action='store_true')
    ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--save-mask', dest='save_mask', action='store_true')
    ap.add_argument('--batch', type=int, default=0, help='0=auto (use snapshot batch)')
    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    exts_set = set([e.strip().lower() for e in args.exts.split(',') if e.strip()])
    files = gather_images(indir, args.recursive, exts_set)
    if not files:
        print(f"[ERR] No images found under {indir} with exts {sorted(exts_set)}")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} images under {indir}")

    # --- VART runner ---
    model = VART.Runner(snapshot_dir=args.snapshot, aie_only=False)
    in_fmt = model.input_shape_formats[0]   # 'NCHW' or 'NHWC'
    out_fmt = model.output_shape_formats[0] # 'NCHW' or 'NHWC'
    in_dtype = model.input_types[0]
    snap_shape = model.input_shapes[0]      # e.g. [4,3,256,256] or [4,256,256,3]
    snap_batch = int(snap_shape[0])

    # 実行バッチは基本スナップショットに合わせる
    req_batch = args.batch if args.batch > 0 else snap_batch
    if req_batch != snap_batch:
        print(f"[WARN] Requested batch {req_batch} != snapshot batch {snap_batch}. "
              f"Snapshot batch {snap_batch} を使用し、足りない分はパディングします。")
    batch = snap_batch

    # 事前確保
    size = args.size
    if in_fmt == 'NCHW':
        x_batch = np.zeros((batch, 3, size, size), dtype=np.float32)
    elif in_fmt == 'NHWC':
        x_batch = np.zeros((batch, size, size, 3), dtype=np.float32)
    else:
        print(f"[ERR] Unsupported input format: {in_fmt}")
        sys.exit(2)

    n_ok, n_ng = 0, 0
    t_all = 0.0

    # バッチごとに処理
    total = len(files)
    for bi in range(0, total, batch):
        chunk = files[bi: bi + batch]
        valid = len(chunk)
        bases = []

        # 前処理 & パック
        for j, img_path in enumerate(chunk):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] skip unreadable: {img_path}")
                n_ng += 1
                continue
            base = letterbox_bgr(img, size)
            bases.append(base)
            s = base.astype(np.float32) / 255.0
            if in_fmt == 'NCHW':
                x_batch[j] = s.transpose(2, 0, 1)
            else:
                x_batch[j] = s

        if valid == 0:
            continue

        # パディング（最後の画像を複製）
        if valid < batch:
            pad_sample = x_batch[valid - 1:valid]
            for j in range(valid, batch):
                x_batch[j] = pad_sample[0]
            # bases も見かけ上合わせておく（保存時は valid 分しか使わない）
            while len(bases) < batch:
                bases.append(bases[-1])

        # 推論
        t0 = time.time()
        out_list = model([x_batch.astype(in_dtype)])
        dt = (time.time() - t0) * 1000.0
        t_all += dt

        y = out_list[0]  # shape: (batch, C,H,W) or (batch,H,W,C)

        # 保存（valid 分だけ）
        for j in range(valid):
            # [1, ...] 形にして overlay
            if out_fmt == 'NCHW':
                logits_1n = y[j:j+1]
            elif out_fmt == 'NHWC':
                logits_1n = y[j:j+1]
            else:
                print(f"[ERR] Unsupported output format: {out_fmt}")
                sys.exit(3)

            overlay, mask_like = overlay_from_logits(
                logits_1n, out_fmt, args.thr, args.alpha, bases[j])

            # 出力パス決定
            img_path = chunk[j]
            rel = img_path.relative_to(indir) if args.mirror else img_path.name
            rel = Path(rel)
            out_png = rel.with_suffix('').name + args.suffix + '.png'
            out_subdir = outdir / (rel.parent if args.mirror else Path())
            ensure_dir(out_subdir)
            out_path = out_subdir / out_png
            cv2.imwrite(str(out_path), overlay)

            if args.save_mask:
                mask_name = rel.with_suffix('').name + '_mask.png'
                mask_path = out_subdir / mask_name
                cv2.imwrite(str(mask_path), mask_like)

            n_ok += 1

        done = min(bi + batch, total)
        print(f"[INFO] {done}/{total} done | last batch {dt:.2f} ms | avg/img {t_all/max(1,n_ok):.2f} ms")

    print(f"[OK] finished. success={n_ok}, failed={n_ng}, avg/img={t_all/max(1,n_ok):.2f} ms")


if __name__ == '__main__':
    main()

