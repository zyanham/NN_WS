python3 -m venv venv
source venv/bin/activate

#mkdir -p original/Auraface-V1
#git clone https://huggingface.co/fal/AuraFace-v1 original/Auraface-V1 

python3 -m pip install --upgrade pip wheel setuptools

# PyTorch (CPU版) + 実行に必要な最低限
pip3 install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip3 install onnx onnxruntime opencv-python pillow numpy==1.23.5 tqdm
pip3 install transformers insightface
