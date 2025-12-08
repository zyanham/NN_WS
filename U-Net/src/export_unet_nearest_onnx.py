# scripts/export_unet_nearest_onnx.py
import argparse, torch, os
from unet_nearest import UNetNearest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights/unet_nearest_256.onnx")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--classes", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    net = UNetNearest(n_channels=3, n_classes=args.classes, base=args.base).eval()

    dummy = torch.randn(1, 3, args.h, args.w)
    torch.onnx.export(
        net, dummy, args.out, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}
    )
    print(f"[OK] Exported ONNX -> {args.out}")

if __name__ == "__main__":
    main()

