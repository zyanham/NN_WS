#!/usr/bin/env python3
import argparse, os, torch, torch.nn as nn

URL_TPL = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale{scale}_epoch2.pth"

def patch_to_nearest(m: nn.Module):
    for name, ch in list(m.named_children()):
        if isinstance(ch, nn.Upsample):
            m.add_module(name, nn.Upsample(scale_factor=ch.scale_factor, mode="nearest"))
        else:
            patch_to_nearest(ch)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights/unet_carvana_nearest_256.onnx")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--scale", type=float, default=0.5, choices=[0.5, 1.0])
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) アーキのみhubから（pretrained=False）
    net = torch.hub.load("milesial/Pytorch-UNet", "unet_carvana",
                         pretrained=False, scale=args.scale, trust_repo=True)
    net.eval()

    # 2) 学習済み重みをCPUへマップして手動ロード
    url = URL_TPL.format(scale=args.scale)
    state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
    _missing, _unexpected = net.load_state_dict(state_dict, strict=False)

    # 3) Upsampleをnearestに差し替え（無重みなので安全）
    patch_to_nearest(net)

    # 4) ONNX出力
    x = torch.randn(1, 3, args.h, args.w)
    torch.onnx.export(
        net, x, args.out, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}
    )
    print(f"[OK] Exported -> {args.out}")

if __name__ == "__main__":
    main()

