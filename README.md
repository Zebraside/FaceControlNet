# stablediffusiontest

# Prerequisites

[RAM model](https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth)

The goal of this project is to train model that can generate controled face expressions

Implementation is based https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py#L541

- [ ] Generate face masks using mediapipe landmarks
- [ ] Finish finetuning pipeline https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
