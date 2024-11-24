from lightning.pytorch.cli import ArgsType, LightningCLI

from lightning_modules.lit_datamodule import FaceConditionDataset
from lightning_modules.lit_facecontrol import LitFaceControl

def cli_main(args: ArgsType = None):
    cli = LightningCLI(
        LitFaceControl,
        FaceConditionDataset,
        args=args
    )
