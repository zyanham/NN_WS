python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools onnxscript

# PyTorch (CPU版) + 実行に必要な最低限
pip3 install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip3 install onnx onnxruntime opencv-python pillow numpy tqdm

