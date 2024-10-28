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

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker_v2_with_blendshapes.task",
                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.IMAGE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = get_transform(image_size=384)
tag_model = ram_plus(pretrained='models/recognize-anything-plus-model/ram_plus_swin_large_14m.pth',
                     image_size=384,
                     vit='swin_l')
tag_model.eval()
model = tag_model.to(device)

with FaceLandmarker.create_from_options(options) as landmarker:
    for imagepath in tqdm.tqdm(images, desc="Processing images", unit="image"):
        mp_image = mp.Image.create_from_file(imagepath)
        pose_landmarker_result = landmarker.detect(mp_image)
        landmarks_dict = {}

        if pose_landmarker_result.face_landmarks:
            landmarks = []
            for landmark in pose_landmarker_result.face_landmarks[0]:
                x = landmark.x
                y = landmark.y
                landmarks.append((x, y))

            landmarks_dict["face_landmarks"] = landmarks
        
        image = transform(Image.open(imagepath)).unsqueeze(0).to(device)
        res = inference(image, tag_model)
        landmarks_dict["tags"] = res[0].replace(" | ", ", ")

        with open(f"data/annotations/{Path(imagepath).name.replace(".jpg", ".json")}", "w") as f:
            json.dump(landmarks_dict, f)

print("Landmarks have been saved to landmarks.json")
