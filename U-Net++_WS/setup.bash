# venv （モデルごとに専用 venv）
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip

# 必要最低限：PyTorch/ONNX/推論・画像IO
pip3 install torch onnx onnxruntime opencv-python pillow numpy onnxscript
