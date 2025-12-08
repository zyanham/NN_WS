#!/usr/bin/env python3
# Snapshot generator (still-image) for Vitis AI 2025.1 (standalone)
import argparse, os, time, importlib
import numpy as np, cv2

def _import_zebra():
    importlib.import_module("zebra_import")
    print("[INFO] zebra_import loaded (snapshot will be generated).")

def _pre(image_path, size_hw):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = size_hw
    if h>0 and w>0: img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32)/255.0).transpose(2,0,1)[None,...]  # (1,3,H,W)
    return img, x

def _save_mask(y_logits, thr, out_path):
    y = 1.0/(1.0+np.exp(-y_logits)); y = y[0,0]
    mask = (y>=thr).astype(np.uint8)*255
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mask)
    print(f"[OK] Saved: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Generate VAISW snapshot (still image).")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--providers", default="CPUExecutionProvider")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    snap_dir = os.path.abspath(args.snap)
    os.makedirs(snap_dir, exist_ok=True)
    os.environ["VAISW_SNAPSHOT_DIRECTORY"] = snap_dir
    os.environ.setdefault("VAISW_RUNSESSION_SUMMARY","all")

    # zebra_import は ORT より先に
    _import_zebra()
    import onnxruntime as ort
    print(f"[INFO] onnxruntime module: {getattr(ort,'__file__','')}")

    providers=[p.strip() for p in args.providers.split(",") if p.strip()]
    sess = ort.InferenceSession(args.onnx, providers=providers)

    _, x = _pre(args.input, (args.h, args.w))
    for _ in range(max(0,args.warmup)): _ = sess.run(None, {"input":x})

    t0=time.time(); y=None
    for _ in range(max(1,args.repeat)):
        y, = sess.run(None, {"input":x})
    print(f"[INFO] Elapsed for repeat={args.repeat}: {time.time()-t0:.3f}s")

    out_path = args.out or os.path.join(snap_dir, "last_mask.png")
    _save_mask(y, args.thr, out_path)
    print(f"[DONE] Snapshot dumped under: {snap_dir}")

if __name__=="__main__":
    main()
