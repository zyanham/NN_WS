#!/usr/bin/env python3
"""
Export the *best* checkpoints from Awesome-U-Net's U-Net++ (ISIC2018 / SegPC2021)
into ONNX files that we can feed to Vitis AI (2025.1).

Usage (from your project root):
  python scripts/export_unetpp_onnx.py \
      --weights-dirs weights/isic2018_unetpp weights/segpc2021_unetpp \
      --size 256 --opset 17

Assumptions
- You have cloned the official Awesome-U-Net repo and (optionally) the UNet++ reference:
    git clone https://github.com/NITR098/Awesome-U-Net third_party/Awesome-U-Net
    git clone https://github.com/4uiiurz1/pytorch-nested-unet third_party/pytorch-nested-unet
- The downloaded checkpoints are the state_dict files like "best_model_state_dict.pt".

Notes
- We try to import a matching model class from Awesome-U-Net, otherwise fall back
  to the popular UNet++ (Nested U-Net) implementation used by that repo.
- If we cannot match *all* keys, we still export with strict=False (best-effort).
- Class count heuristics: ISIC2018→1 class (binary), SegPC2021→3 classes (bg/cyto/nucleus).
  If result.json in the weights dir defines a different number, it takes precedence.
"""
import argparse
import json
import os
import sys
import inspect
from pathlib import Path

import torch
import onnx

# -------------- Utils --------------

def add_third_party_paths(project_root: Path):
    """Add likely third_party paths for Awesome-U-Net and Nested U-Net.
    Also add the Awesome-U-Net/models subdir so `from models import ...` works.
    """
    tp = project_root / "third_party"
    candidates = [
        tp / "Awesome-U-Net",
        tp / "Awesome-U-Net" / "models",
        tp / "pytorch-nested-unet",
    ]
    for p in candidates:
        if p.exists():
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
                print(f"[INFO] sys.path += {sp}")


def read_result_json(weights_dir: Path):
    cfg = {}
    rj = weights_dir / "result.json"
    if rj.exists():
        try:
            cfg = json.loads(rj.read_text())
        except Exception:
            pass
    return cfg


def guess_num_classes(weights_dir: Path, cfg: dict) -> int:
    for k in ("num_classes", "n_classes", "classes"):
        if k in cfg and isinstance(cfg[k], int) and cfg[k] > 0:
            return cfg[k]
    name = weights_dir.name.lower()
    if "segpc" in name:
        return 3  # bg, cytoplasm, nucleus
    return 1  # ISIC2018 (lesion vs bg)


def guess_in_ch(cfg: dict) -> int:
    for k in ("in_ch", "in_channels", "input_channels"):
        if k in cfg and isinstance(cfg[k], int) and cfg[k] > 0:
            return cfg[k]
    return 3


def make_dummy_input(nc: int, size: int) -> torch.Tensor:
    return torch.randn(1, nc, size, size, dtype=torch.float32)


# -------------- Model Resolver --------------
class ModelResolver:
    """Try multiple known U-Net++ implementations and pick the first that works."""

    def __init__(self):
        self.trials = [
            ("Awesome-UNet.models.unetpp.UNetPP", self._try_awesome_unetpp),
            ("Awesome-UNet.models.unetplusplus.UNetPlusPlus", self._try_awesome_unetplusplus),
            ("pytorch-nested-unet.archs.NestedUNet", self._try_nested_unet),
            ("archs.NestedUNet", self._try_nested_unet),  # if repo is at sys.path root
        ]

    def resolve(self, in_ch: int, classes: int, cfg: dict):
        last_err = None
        for name, fn in self.trials:
            try:
                model = fn(in_ch, classes, cfg)
                return model, name
            except Exception as e:
                last_err = e
        raise RuntimeError(
            f"Could not construct a UNet++ model from known sources. Last error: {last_err}"
        )

    # --- concrete builders ---
    def _try_awesome_unetpp(self, in_ch: int, classes: int, cfg: dict):
        from models.unetpp import UNetPP  # type: ignore
        return self._instantiate_by_signature(UNetPP, in_ch, classes, cfg)

    def _try_awesome_unetplusplus(self, in_ch: int, classes: int, cfg: dict):
        from models.unetplusplus import UNetPlusPlus  # type: ignore
        return self._instantiate_by_signature(UNetPlusPlus, in_ch, classes, cfg)

    def _try_nested_unet(self, in_ch: int, classes: int, cfg: dict):
        # 4uiiurz1 implementation often exposes NestedUNet(in_ch=3, out_ch=1, **kw)
        try:
            from archs import NestedUNet  # type: ignore
        except Exception:
            from pytorch_nested_unet.archs import NestedUNet  # alternate package name
        return self._instantiate_by_signature(NestedUNet, in_ch, classes, cfg)

    @staticmethod
    def _instantiate_by_signature(Cls, in_ch: int, classes: int, cfg: dict):
        sig = inspect.signature(Cls)
        kwargs = {}
        # common knobs across various repos
        mapping = {
            "in_ch": in_ch,
            "in_channels": in_ch,
            "input_channels": in_ch,
            "out_ch": classes,
            "num_classes": classes,
            "n_classes": classes,
            "classes": classes,
            "deep_supervision": cfg.get("deep_supervision", False),
            "is_deep_supervision": cfg.get("deep_supervision", False),
        }
        for k, v in mapping.items():
            if k in sig.parameters:
                kwargs[k] = v
        try:
            return Cls(**kwargs)
        except TypeError:
            # Some repos require positional args (in_ch, out_ch, ...)
            try:
                return Cls(in_ch, classes)
            except Exception:
                # Last resort: no-arg
                return Cls()


# -------------- Main --------------

def _load_state_dict(weights_dir: Path):
    best = weights_dir / "best_model_state_dict.pt"
    last = weights_dir / "last_model_state_dict.pt"
    sd_path = best if best.exists() else last
    if not sd_path.exists():
        raise FileNotFoundError(f"No state_dict found under {weights_dir}")
    sd = torch.load(str(sd_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    # strip DataParallel prefix if present
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _detect_classes_from_sd(sd: dict, default: int) -> int:
    candidates = [
        "final.weight",
        "outc.weight",
        "classifier.weight",
        "last_conv.weight",
        "conv_final.weight",
        "segmentation_head.weight",
        "segmentation_head.0.weight",
    ]
    for k in candidates:
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) >= 1:
            try:
                return int(sd[k].shape[0])
            except Exception:
                pass
    # heuristic fallback: look for any conv "*.weight" ending with "final" in name
    for k, v in sd.items():
        if k.endswith(".weight") and ("final" in k or "outc" in k or "head" in k):
            try:
                return int(v.shape[0])
            except Exception:
                continue
    return default


def _filtered_load(model: torch.nn.Module, sd: dict):
    msd = model.state_dict()
    ok, skip = 0, 0
    picked = {}
    for k, v in sd.items():
        if k in msd and hasattr(v, "shape") and hasattr(msd[k], "shape"):
            if tuple(v.shape) == tuple(msd[k].shape):
                picked[k] = v
                ok += 1
            else:
                skip += 1
        else:
            skip += 1
    missing, unexpected = model.load_state_dict(picked, strict=False)
    print(f"[INFO] loaded tensors: {ok}, skipped (shape/name): {skip}, missing after load: {len(missing)}, unexpected: {len(unexpected)}")


def export_one(weights_dir: Path, out_dir: Path, size: int, opset: int):
    cfg = read_result_json(weights_dir)
    in_ch = guess_in_ch(cfg)

    # read sd first so we can auto-detect class count from the checkpoint
    sd = _load_state_dict(weights_dir)
    classes_default = guess_num_classes(weights_dir, cfg)
    classes = _detect_classes_from_sd(sd, classes_default)
    if classes != classes_default:
        print(f"[INFO] num_classes overridden by checkpoint: {classes_default} -> {classes}")

    # prepare model
    resolver = ModelResolver()
    model, source = resolver.resolve(in_ch, classes, cfg)
    print(f"[INFO] Model source: {source}")

    # try loading with filtering to avoid size-mismatch crashes
    _filtered_load(model, sd)
    model.eval()

    # dummy input
    x = make_dummy_input(in_ch, size)

    # export
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = weights_dir.name.replace("_unetpp", "")
    onnx_path = out_dir / f"unetpp_{tag}_best_{size}.onnx"

    torch.onnx.export(
        model,
        x,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"[OK] Exported: {onnx_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dirs", nargs="+", required=True,
                    help="List of weights directories to export (each contains best_model_state_dict.pt)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out-dir", default="weights")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    add_third_party_paths(project_root)

    out_dir = Path(args.out_dir)
    for wd in args.weights_dirs:
        export_one(Path(wd), out_dir, args.size, args.opset)


def main():
    pass  # placeholder to keep regex anchor; real main defined above

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dirs", nargs="+", required=True,
                    help="List of weights directories to export (each contains best_model_state_dict.pt)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out-dir", default="weights")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    add_third_party_paths(project_root)

    out_dir = Path(args.out_dir)
    for wd in args.weights_dirs:
        export_one(Path(wd), out_dir, args.size, args.opset)


if __name__ == "__main__":
    # call the real main (defined above)
    main()

