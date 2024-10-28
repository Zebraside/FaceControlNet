import math
import logging

import torch
import torch.nn.functional as F
import lightning as L

from packaging import version
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel,UNet2DModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, PretrainedConfig

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

class LitFaceControl(L.LightningModule):
    def __init__(self,
                 pretrained_model_path: str,
                 controlnet_model_or_path: str|None = None,
                 lr: float = 1e-3,
                 adam_beta1: float = 0.9,
                 adam_beta2: float = 0.999,
                 adam_weight_decay: float = 1e-2,
                 adam_epsilon: float = 1e-8,
                 lr_scheduler: str = "constant",
                 lr_warmup_steps: int = 0,
                 noise_offset: float = 0.0,
                 sne_gamma: float = None,
                 use_xformers = True,
                 revision = None,
                 variant = None,
                 enable_gradient_checkpointing = False,
                 prediction_type=None,
                 num_training_steps_scheduler=None,
                 allow_tf32=False,
                 scale_lr=False,
                 use_8bit_adam=False):
        super().__init__()
        self.save_hyperparameters()

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", revision=revision)
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", revision=revision, variant=variant)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", revision=revision, variant=variant)
        text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_path, revision)
        self.text_encoder = text_encoder_cls.from_pretrained(
            pretrained_model_path, subfolder="text_encoder", revision=revision, # variant=args.variant
        )

        if controlnet_model_or_path:
            logging.info("Loading controlnet model")
            self.controlnet = ControlNetModel.from_pretrained(controlnet_model_or_path)
        else:
            self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=4)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        # self.tokenizer.requires_grad_(False)
        self.unet.requires_grad_(False)

        if use_xformers:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    print("xFormers 0.0.16 cannot be used for training in some GPUs.")
                    # logger.warning(
                    #     "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    # )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly.")

        if enable_gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()

        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if scale_lr:
            # TODO: copy this from the original code
            # args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            pass

        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            self.optimizer_class = bnb.optim.AdamW8bit
        else:
            self.optimizer_class = torch.optim.AdamW

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        latents = self.vae.encode(batch["image"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        # zeros_channel = torch.zeros(latents.size(0), 1, latents.size(2), latents.size(3), device=latents.device)
        # latents = torch.cat((latents, zeros_channel), dim=1)

        noise = torch.rand_like(latents)
        if self.hparams.noise_offset:
            noise = self.hparams.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        bsz = latents.shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz, ), device=latents.device).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        tokens = self.tokenizer(batch["text"], return_dict=False, return_tensors="pt")['input_ids'].to(latents.device)
        encoder_hidden_states = self.text_encoder(tokens, return_dict=False)[0]
        controlnet_image = torch.concat([batch["ref_image"], batch["face_mask"]] , dim=1)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[
                sample for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        if self.hparams.prediction_type is not None:
            self.noise_scheduler.register_to_config(prediction_type=self.hparams.prediction_type)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss


    @property
    def num_training_steps_for_scheduler(self) -> int:
        if self.trainer.max_steps is None: # TODO: check if it is none by default
            # TODO: check if num_training_batches is valid
            len_train_dataloader_after_sharding = math.ceil(len(self.trainer.num_training_batches * self.hparams.batch_size) / self.trainer.num_devices)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / self.trainer.accumulate_grad_batches)
            num_training_steps_for_scheduler = self.trainer.max_epochs * num_update_steps_per_epoch * self.trainer.num_devices
        else:
            num_training_steps_for_scheduler = self.trainer.max_steps * self.trainer.num_devices
        return num_training_steps_for_scheduler

    @property
    def num_warmup_steps_for_scheduler(self) -> int:
        return self.hparams.lr_warmup_steps * self.trainer.num_devices

    @property
    def lr(self) -> float:
        return self.hparams.lr * self.trainer.num_devices

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.controlnet.parameters(),
                                         lr=self.hparams.lr,
                                         betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                                         weight_decay=self.hparams.adam_weight_decay,
                                         eps=self.hparams.adam_epsilon
                                         )


        scheduler = get_scheduler(
            self.hparams.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps_for_scheduler,
            num_training_steps=self.num_training_steps_for_scheduler
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }