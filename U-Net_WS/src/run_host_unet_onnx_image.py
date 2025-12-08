# scripts/run_host_unet_onnx_image.py
import argparse, os, cv2, numpy as np, onnxruntime as ort

def load_image(path, size_hw):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = size_hw
    if h>0 and w>0: img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))[None, ...]  # NCHW
    return img, x

def save_mask(mask01, out_path):
    mask_u8 = (mask01 * 255).astype(np.uint8)
    cv2.imwrite(out_path, mask_u8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results/mask.png")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5, help="sigmoid後の閾値")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # CPU ExecutionProvider でOK（ホスト検証）
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    _, x = load_image(args.input, (args.h, args.w))
    y, = sess.run(None, {"input": x})

    # 出力はロジット想定→シグモイド→閾値化
    y = 1.0 / (1.0 + np.exp(-y))
    y = y[0, 0]  # (1,1,H,W)
    mask = (y >= args.thr).astype(np.float32)

    save_mask(mask, args.out)
    print(f"[OK] Saved: {args.out}")

if __name__ == "__main__":
    main()

