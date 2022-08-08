import pytorch_lightning as pl
from batchgenerators.utilities.file_and_folder_operations import join, isfile
import os
from typing import Dict, Any
from pytorch_lightning import Callback


class nnUNetCheckpoint(Callback):
    def __init__(self, alpha: float = 0.9, save_every: int = 50):
        super().__init__()

        self.alpha = alpha
        self.save_every = save_every
        self._best_ema = None
        self._current_ema = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            '_best_ema': self._best_ema,
            '_current_ema': self._current_ema,
            'alpha': self.alpha,
            'save_every': self.save_every,

        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._best_ema = state_dict['_best_ema']
        self._current_ema = state_dict['_current_ema']
        self.alpha = state_dict['alpha']
        self.save_every = state_dict['save_every']

    # def on_load_checkpoint(
    #     self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    # ) -> None:
    #     self._best_ema = callback_state['_best_ema']
    #     self._current_ema = callback_state['_current_ema']
    #     self.alpha = callback_state['alpha']
    #     self.save_every = callback_state['save_every']

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        current_epoch = pl_module.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (pl_module.num_epochs - 1):
            trainer.save_checkpoint(join(pl_module.output_folder, 'checkpoint_latest.pth'), False)
        elif current_epoch == pl_module.num_epochs - 1:
            trainer.save_checkpoint(join(pl_module.output_folder, 'checkpoint_final.pth'), True)
            # delete latest checkpoint
            if isfile(join(pl_module.output_folder, 'checkpoint_latest.pth')):
                os.remove(join(pl_module.output_folder, 'checkpoint_latest.pth'))

        # exponential moving average of pseudo dice to determine 'best' model (use with caution)
        self._current_ema = self._current_ema * 0.9 + 0.1 * pl_module.my_fantastic_logging['mean_fg_dice'][-1] \
            if self._current_ema is not None else pl_module.my_fantastic_logging['mean_fg_dice'][-1]
        if self._best_ema is None or self._current_ema > self._best_ema:
            pl_module.print_to_log_file(f"New best checkpoint! Yayy!")
            self._best_ema = self._current_ema
            trainer.save_checkpoint(join(pl_module.output_folder, 'checkpoint_best.pth'), False)
