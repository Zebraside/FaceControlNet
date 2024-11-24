# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py#L541
# TODO: add scalr_lr parameter
import argparse
import math

from lightning_modules.lit_datamodule import LitDataModule
from lightning_modules.lit_facecontrol import LitFaceControl
import lightning as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    args = parser.parse_args()


    dataloaders = LitDataModule(train_data_dir="data/img_align_celeba/img_align_celeba",
                                train_ann_dir="data/annotations",
                                train_attributes_path="data/list_identity_celeba.csv",
                                train_facelandmarks_path="data/face_landmarks.json",)


    model = LitFaceControl(pretrained_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5")

    trainer = pl.Trainer(precision=16,
                         accelerator="gpu",
                         max_epochs=1,
                         )

    trainer.fit(model=model,
               datamodule=dataloaders)
