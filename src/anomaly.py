from pathlib import Path
from distutils.version import StrictVersion

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load

from hydra.utils import instantiate

from src.utils.callbacks import LitTQDMProgressBar
from src.utils.callbacks import LitProgressBar

class AnomalyDetector(object):
    """Class for anomaly detection experiments"""
    def __init__(self, cfg, overrides:dict=None):
        """
        Args:
            config (dict): config
        """
        self.config = cfg
        self.logger = None
        self.early_stopping  = None
        self.model_checkpoint = None
        
        self.init_model()
        
        # logger instance の生成
        self.logger = instantiate(cfg.logger)
        # pl 1.7.0 の時点で progressbar の callback が変わってしまったため削除
        # jupyter だと validation の progressbar で無駄な改行が発生を削るため
        #if 'progress_bar_refresh_rate' in self.config.trainer.args:
        #    self.progressbar = MyProgressBar(self.config.trainer.args.progress_bar_refresh_rate)
        #else:
        #    self.progressbar = MyProgressBar()
        self.progressbar = instantiate(cfg.callbacks.ProgressBar)

        # trainer 用 instance の生成
        self.early_stopping = None
        if self.config.trainer.use_early_stopping:
            self.early_stopping = instantiate(cfg.callbacks.EarlyStopping)
        
        save_dir = cfg.logger.save_dir if self.logger.save_dir is None else self.logger.save_dir
        model_log_dir = str(self._get_model_path(save_dir, self.logger.experiment_id, self.logger.run_id))
        if 'dirpath' in self.config.callbacks.ModelCheckpoint:
            self.config.callbacks.ModelCheckpoint.dirpath = model_log_dir
        self.model_checkpoint = instantiate(cfg.callbacks.ModelCheckpoint)
    
    def _get_model_path(self, save_dir, experiment_id=None, run_id=None):
        path = Path(save_dir)/experiment_id/run_id/'artifacts'/'models'
        return path
        
        
    def init_model(self):
        """Initialize the model"""
        # model インスタンスの生成
        exec(f'from .models import {self.config.model.name}')
        #self.model = instantiate(cfg.model.instance, cfg=cfg.model)
        self.model = eval(self.config.model.name)(self.config.model)
        print(self.model)
        
        
    def create_trainer(self, max_epochs=2):
        
        if max_epochs is not None:
            self.config.trainer.args.max_epochs = max_epochs
        
        if torch.cuda.is_available() and self.config.trainer.use_gpu:
            if 'gpus' in self.config.trainer.args:
                self.config.trainer.args.gpus = 1 # ver 2 以降はなくなるらしい
            else:
                self.config.trainer.args.accelerator = 'gpu'
                self.config.trainer.args.devices = 1
        else:
            if 'gpus' in self.config.trainer.args:
                self.config.trainer.args.gpus = 0
            else:
                self.config.trainer.args.accelerator = 'cpu'
        
        callbacks=[self.progressbar]
        if self.early_stopping is not None:
            callbacks.append(self.early_stopping)
        if self.model_checkpoint is not None:
            callbacks.append(self.model_checkpoint)
        
        self.trainer = pl.Trainer(logger=self.logger, callbacks=callbacks, **(self.config.trainer.args))
        
    def train(self, train_dataloader=None, val_dataloader=None, dm=None, step=0, max_epochs=None):
        """Train

        Args:
            step (int): from which step to start to train
            num_epochs (int): how many epochs for training
            best_metric (int): the best metric before training
        """
        # Avoid overwriting existing checkpoints in train mode
        # if step == 0 and self.get_checkpoint_path(step).exists():
        #     raise FileExistsError(f'{self.get_checkpoint_path(step)} has already exists. '
        #         f'Please use other config file (.yml) or remove {self.get_checkpoint_path(step)}.')
        #self.model.init_ema()  # initialize Exponential Moving Average
        self.create_trainer(max_epochs)

        self.trainer.fit(self.model, dm)
    
    def load_ckpt(self, n_epoch, 
                  logger=None, 
                  save_dir=None, experiment_id=None, run_id=None):
        """restart to train the model from step for num_epochs"""
        if self.model.anomaly_scores is not None:
            self.model.anomaly_scores = None
        if self.model.recon_x is not None:
            self.model.recon_x = None
        
        if logger is None:
            logger = self.logger
         
        if not (run_id is None):
            folder_path = self._get_model_path(save_dir, experiment_id, run_id)
        else:
            # mlflow が db を使っていると save_dir が None になるため
            save_dir = self.config.logger.save_dir if self.logger.save_dir is None else self.logger.save_dir
            folder_path = self._get_model_path(save_dir, logger.experiment_id, logger.run_id)
        
        if n_epoch == 'last':
            if (folder_path/'last.ckpt').exists():
                checkpoint_file = 'last.ckpt'
            else:
                n_epoch = self.get_last_step(folder_path)
                checkpoint_file = f'model-epoch={n_epoch}.ckpt'
        else:
            checkpoint_file = f'model-epoch={n_epoch}.ckpt'
            if n_epoch > self.config.trainer.args.max_epochs:
                self.config.trainer.args.max_epochs = n_epoch + 1

        # 本当は n_epoch が best かどうかを判定したほうがよさそう
        ckpt_path = str(folder_path/checkpoint_file)
        if run_id is None and self.model_checkpoint.best_model_path is not None:
            if len(self.model_checkpoint.best_model_path) != 0:
                ckpt_path = self.model_checkpoint.best_model_path
        
        '''
        ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
        print(ckpt)
        self.model.load_state_dict(ckpt['state_dict'])
        
        if torch.cuda.is_available() and self.config.trainer.use_gpu:
            if 'gpus' in self.config.trainer.args:
                self.config.trainer.args.gpus = 1 # ver 2 以降はなくなるらしい
            else:
                self.config.trainer.args.accelerator = 'gpu'
                self.config.trainer.args.devices = 1
        else:
            if 'gpus' in self.config.trainer.args:
                self.config.trainer.args.gpus = 0
            else:
                self.config.trainer.args.accelerator = 'cpu'
        
        callbacks=[self.progressbar]
        if self.early_stopping is not None:
            callbacks.append(self.early_stopping)
        if self.model_checkpoint is not None:
            callbacks.append(self.model_checkpoint)
        
        print(self.config.trainer.args)
        self.trainer = pl.Trainer(resume_from_checkpoint=ckpt_path,
                                  logger=self.logger, callbacks=callbacks, **(self.config.trainer.args))
        '''
        return ckpt_path

    def train_from(self, n_epoch, max_epochs=None, 
                   logger=None, 
                   save_dir=None, experiment_id=None, run_id=None,
                   train_dataloader=None, val_dataloader=None, dm=None):
        
        #if max_epochs is not None:
        #    self.config.trainer.args.max_epochs = max_epochs
        
        ckpt_path = self.load_ckpt(n_epoch, logger, save_dir, experiment_id, run_id)
        
        if not hasattr(self,'trainer'):
            self.create_trainer(max_epochs)

        if dm is not None:
            self.trainer.fit(self.model, datamodule=dm, ckpt_path = ckpt_path)
        else:
            self.trainer.fit(self.model, train_dataloader, val_dataloader, ckpt_path = ckpt_path)
        #metric, best_metric = self.load_weight(step)
        #self.train(train_dataloader, val_dataloader, step=step, num_epochs=num_epochs, best_metric=best_metric)

    def val(self, val_dataloader):
        """Compute loss and metrics on val set"""
        pass

    def test(self, test_dataloader=None, dm=None, max_epochs=None):
        self.model.anomaly_scores = None if self.model.anomaly_scores is not None else None
        self.model.recon_x = None if self.model.recon_x is not None else None
        """Compute loss and metrics on test set"""
        # Trainer を resume したとき、なぜか test も resume して行おうとするため、以下の処理を追加... なんでこんな仕様？
        
        # need to set current number of epochs under the trainer.max_epochs 
        if max_epochs is not None:
            if self.trainer.max_epochs < max_epochs:
                self.trainer.max_epochs = max_epochs
                
        if test_dataloader is not None:
            self.trainer.test(self.model, test_dataloaders=test_dataloader)
        elif dm is not None:
            self.trainer.test(self.model, datamodule=dm)
    
    def get_anomaly_scores(self, dataloader=None, dm=None):
        '''
        Returns
        -------------------------
        anomaly_score list(batch_size, features)
        '''
        if self.model.anomaly_scores is None:
            self.test(dataloader, dm)
        return self.model.anomaly_scores
    
    def generate(self, dataloader=None, dm=None):
        if self.model.recon_x is None:
            self.test(dataloader, dm)
        return self.model.recon_x
    
    def reset_score(self):
        self.model.anomaly_scores = None
        self.model.recon_x = None
        
    def predict(self, dataloader=None, dm=None):
        """Make prediction from the specified file
        Args:
            dataloader (torch.utils.data.DataLoader):
        Return:
            pred (np.ndarray): prediction
        """
        pass    

    def get_last_step(self, ckpt_path):
        """Get the largest step in a checkpoint directory.

        Return:
            step (int):
        """
        files = ckpt_path.glob('*.ckpt')
        steps = [re.findall(f'model-epoch=(\d+).pt', f.name) for f in files]
        step = max(map(lambda y: int(y[0]), filter(lambda x: len(x) > 0, steps)))
        return step