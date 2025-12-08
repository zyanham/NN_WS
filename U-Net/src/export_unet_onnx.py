# scripts/export_unet_onnx.py
import argparse, torch, os

URL_TPL = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale{scale}_epoch2.pth"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights/unet_carvana_256.onnx")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--scale", type=float, default=0.5, choices=[0.5, 1.0],
                    help="torch.hubのCarvana学習済みと同じスケール（0.5 or 1.0）")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="未学習でエクスポート（デバッグ用）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) モデル定義のみhubから取得（重みは後で手動ロード）
    net = torch.hub.load(
        'milesial/Pytorch-UNet', 'unet_carvana',
        pretrained=False, scale=args.scale, trust_repo=True
    )
    net.eval()

    # 2) 学習済み重み（CPUへマップして）を手動ロード
    if not args.no_pretrained:
        url = URL_TPL.format(scale=args.scale)
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=True, map_location='cpu'
        )
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded weights from {url}")
        if missing:   print(f"[WARN] Missing keys: {missing}")
        if unexpected:print(f"[WARN] Unexpected keys: {unexpected}")

    # 3) ダミー入力でONNXへ
    dummy = torch.randn(1, 3, args.h, args.w)
    torch.onnx.export(
        net, dummy, args.out, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}
    )
    print(f"[OK] Exported ONNX -> {args.out}")

if __name__ == "__main__":
    main()

