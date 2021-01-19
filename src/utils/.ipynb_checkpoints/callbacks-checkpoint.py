from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress import ProgressBar

import sys
import importlib
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

class MyProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        
    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + 1),
            disable=True,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar