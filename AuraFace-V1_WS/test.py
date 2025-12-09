## 動作確認用
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
import numpy as np
import cv2

snapshot_download(
    "fal/AuraFace-v1",
    local_dir="models/auraface",
)
face_app = FaceAnalysis(
    name="auraface",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    root=".",
)

input_image = cv2.imread("./Dataset/HumanFaces/001.jpg")

cv2_image = np.array(input_image.convert("RGB"))

cv2_image = cv2_image[:, :, ::-1]
faces = face_app.get(cv2_image)
embedding = faces[0].normed_embedding

