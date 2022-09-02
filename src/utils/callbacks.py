from pytorch_lightning import Callback
#from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.callbacks import TQDMProgressBar

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
        # 個々が問題ぽい。ロガーとの関連が変わったのかな？
        self.metrics.append(trainer.callback_metrics)

class LitTQDMProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation...')
        return bar

class LitProgressBar(ProgressBarBase):
    def __init__(self):
        pass
    def disable(self):
        self.enable = False
    
    def get_metrics(self, trainer, model):
        items = super.get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_batch_end(self, trainer, pl_module, outputs):
        super().on_train_batch_end(trainer, pl_module, outputs)
        percent = (self.on_train_batch_idx / self.on_train_batch_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')


'''
class MyProgressBar(ProgressBar):
    def __init__(self, *args, **kwargs):
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
'''