import glob
from pathlib import Path
import mediapipe as mp
from PIL import Image, ImageDraw
import json
import tqdm
import pandas as pd

import torch

from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

images = glob.glob("data/img_align_celeba/img_align_celeba/*.jpg")
landmarks_dict = {path: {} for path in images}

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker_v2_with_blendshapes.task",
                             delegate=mp.tasks.BaseOptions.Delegate.GPU),
    running_mode=VisionRunningMode.IMAGE)

with FaceLandmarker.create_from_options(options) as landmarker:
    for imagepath in tqdm.tqdm(images, desc="Processing image landmarks", unit="image"):
        mp_image = mp.Image.create_from_file(imagepath)
        pose_landmarker_result = landmarker.detect(mp_image)

        if pose_landmarker_result.face_landmarks:
            landmarks = []
            for landmark in pose_landmarker_result.face_landmarks[0]:
                x = round(landmark.x, 2)
                y = round(landmark.y, 2)
                landmarks.append((x, y))

            landmarks_dict[imagepath]["face_landmarks"] = landmarks



device = "cuda"
transform = get_transform(image_size=384)
tag_model = ram_plus(pretrained='models/recognize-anything-plus-model/ram_plus_swin_large_14m.pth',
                     image_size=384,
                     vit='swin_l')
tag_model.eval()
model = tag_model.to(device)

for imagepath in tqdm.tqdm(images, desc="Processing tags", unit="image"):
        image = transform(Image.open(imagepath)).unsqueeze(0).to(device)
        res = inference(image, tag_model)
        landmarks_dict[imagepath]["tags"] = res[0].replace(" | ", ", ")

for path, data in tqdm.tqdm(landmarks_dict.items(), desc="Creating annotations", unit="image"):
    with open(f"data/annotations/{Path(path).name.replace('.jpg', '.json')}", "w") as f:
        json.dump(landmarks_dict, f)
