#!/usr/bin/env python3
import onnx
from onnx import helper, numpy_helper

def get_init(m, name):
    for t in m.graph.initializer:
        if t.name == name: return t
    return None

def add_const(m, name, np_val):
    t = numpy_helper.from_array(np_val, name=name)
    m.graph.initializer.extend([t])
    return t

def force_resize_nearest(m):
    # すべてのResizeを: inputs = [X, roi(empty), scales, (sizesなし)] に統一
    import numpy as np
    for n in m.graph.node:
        if n.op_type != "Resize": continue
        # 属性をnearest/asymmetric/floorに
        def set_attr(name,val):
            for a in n.attribute:
                if a.name==name:
                    a.s = val.encode("utf-8")
                    break
            else:
                n.attribute.extend([helper.make_attribute(name, val)])
        set_attr("mode","nearest")
        set_attr("coordinate_transformation_mode","asymmetric")
        set_attr("nearest_mode","floor")
        # 入力を整理: X, roi, scales の3つ構成に
        # 典型UNetは倍化(×2)なので scales=[1,1,2,2]
        # 既存scalesがなければ追加
        if len(n.input) < 3 or n.input[2] == "":
            scales_name = n.name + "_scales"
            add_const(m, scales_name, np.array([1.,1.,2.,2.], dtype=np.float32))
            # roiは空テンソル
            roi_name = n.name + "_roi"
            add_const(m, roi_name, np.array([], dtype=np.float32))
            # 入力を [X, roi, scales] に詰め替え（sizesは捨てる）
            x = n.input[0]
            n.input[:] = [x, roi_name, scales_name]
        else:
            # 既存にsizesがあれば削除（第4入力）
            while len(n.input) > 3:
                n.input.pop()
    return m

def main(inp, outp):
    m = onnx.load(inp)
    m = force_resize_nearest(m)
    onnx.checker.check_model(m)
    onnx.save(m, outp)
    print(f"[OK] Patched Resize -> {outp}")

if __name__ == "__main__":
    import argparse, numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()
    main(args.inp, args.outp)

