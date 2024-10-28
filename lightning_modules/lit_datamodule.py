import json
from pathlib import Path

import pandas as pd
from PIL import Image
import numpy as np

from torch.utils.data import DataLoader, Dataset
import lightning as L
import albumentations as alb
from albumentations.pytorch import ToTensorV2

class FaceConditionDataset(Dataset):
    def __init__(self, data_dir: str, ann_dir: str, attributes_path: str, facelandmarks_path: str, image_transforms):
        self.data_dir = Path(data_dir)
        self.ann_dir = Path(ann_dir)
        self.image_list = pd.read_csv(attributes_path)

        self.transforms = image_transforms

    @staticmethod
    def __create_mask(landmarks, width, height):
        mask = Image.new("L", (width, height), 0)
        for landmark in landmarks:
            mask.putpixel((int(landmark[0] * width), int(landmark[1] * height)), 255)
        mask = np.reshape(np.array(mask), (height, width, 1))
        return mask

    def __get_image_for_id(self, identity_id):
        image_name = self.image_list.groupby("identity_id").get_group(identity_id).sample(1)["image_id"].values[0]
        return np.array(Image.open(self.data_dir / image_name))

    def __getitem__(self, item_idx):
        image_name = self.image_list.iloc[item_idx]["image_id"]
        image =  np.array(Image.open(self.data_dir / image_name))
        with open(self.ann_dir / image_name.replace(".jpg", ".json")) as f:
            ann = json.load(f)
            landmarks = ann["face_landmarks"]
            text = ann["tags"]

        mask =  np.array(self.__create_mask(landmarks, image.shape[1], image.shape[0]))
        transforms = self.transforms(image=image, mask=mask)
        image, mask = transforms['image'] / 255, transforms['mask']
        mask = mask.permute((2, 0, 1))

        reference_image = self.__get_image_for_id(self.image_list.iloc[item_idx]["identity_id"])
        reference_image = self.transforms(image=reference_image)['image'] / 255
        return {
            "image": image,
            "face_mask": mask,
            "ref_image": reference_image,
            "text": text
        }

    def __len__(self):
        return len(self.image_list)


class LitDataModule(L.LightningDataModule):
    def __init__(self, train_data_dir: str,
                       train_ann_dir: str,
                       train_attributes_path: str,
                       train_facelandmarks_path: str,
                       batch_size: int = 1):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.train_attributes_path = train_attributes_path
        self.train_facelandmarks_path = train_facelandmarks_path
        self.train_ann_dir = train_ann_dir

        self.batch_size = batch_size

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def setup(self, stage: str):
        train_transforms = alb.Compose([
            alb.Resize(width=256, height=256),
            ToTensorV2()])

        self.train_dataset = FaceConditionDataset(self.train_data_dir,
                                                  self.train_ann_dir,
                                                  self.train_attributes_path,
                                                  self.train_facelandmarks_path,
                                                  image_transforms=train_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=7, persistent_workers=True)