from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

class ProgressPngCallback(Callback):
    def __init__(self, smooth: float = 0.9):
        self.smooth = smooth
        super().__init__()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

