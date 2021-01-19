import os
import sys
from pathlib import Path
from hydra.utils import instantiate

import torch
import pytorch_lightning as pl

import logging
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.utils.callbacks import MetricsCallback, MyProgressBar

class AD_ParametersSearch(object):
    def __init__(self, cfg, dm):
        self.config = cfg
        self.dm = dm
        pruner = optuna.pruners.MedianPruner()
        if 'storage' in self.config.optuna.study:
            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        self.study = optuna.create_study(pruner=pruner, **self.config.optuna.study, load_if_exists=True)
    
    def do_optimize(self):
        self.study.optimize(self.objective, **self.config.optuna.optimize)
    
    def resume_study(self, n_trials=100):
        self.study = optuna.create_study(**self.config.optuna.study, load_if_exists=True)
        self.study.optimize(self.objective, n_trials=n_trials)
    
    def objective(self, trial):
        
        self.trial = trial
        
        # logger instance の生成 logger = None をしたいが、callbacks の list に None を入れれないので困った...
        self.logger = instantiate(self.config.logger)
        
        # trainer 用 instance の生成
        self.early_stopping = None
        if self.config.trainer.use_early_stopping:
            self.early_stopping = instantiate(self.config.callbacks.EarlyStopping)
        
        model_log_dir = str(self._get_model_path(self.logger.save_dir, self.logger.experiment_id, self.logger.run_id))
        if 'dirpath' in self.config.callbacks.ModelCheckpoint:
            self.config.callbacks.ModelCheckpoint.dirpath = model_log_dir
        self.model_checkpoint = instantiate(self.config.callbacks.ModelCheckpoint) 
        
        self.metrics_callback = MetricsCallback()
        self.progressbar = MyProgressBar()
        self.create_trainer()
        self.init_model()
        
        self.trainer.fit(self.model, self.dm)
        
        return self.metrics_callback.metrics[-1]['val_loss'].item()
        
        
    def init_model(self):
        """Initialize the model"""
        # model インスタンスの生成
        exec(f'from .models import {self.config.model.name}_PS')
        #self.model = instantiate(cfg.model.instance, cfg=cfg.model)
        self.model = eval(f'{self.config.model.name}_PS')(self.config.model, self.trial)
        #print(self.model)       
        
        
    def create_trainer(self):

        if torch.cuda.is_available() and self.config.trainer.use_gpu:
            self.config.trainer.args.gpus = 1
        else:
            self.config.trainer.args.gpus = 0
        
        if self.early_stopping is not None:
            self.trainer = pl.Trainer(logger=self.logger,
                                      callbacks=[self.early_stopping, self.model_checkpoint, self.progressbar,
                                                 self.metrics_callback, PyTorchLightningPruningCallback(self.trial, monitor='val_loss')],
                                      **(self.config.trainer.args))
        else:
            self.trainer = pl.Trainer(logger=self.logger,
                                      callbacks=[self.model_checkpoint, self.progressbar,
                                                 self.metrics_callback, PyTorchLightningPruningCallback(self.trial, monitor='val_loss')], 
                                      **(self.config.trainer.args))    
    
    def _get_model_path(self, save_dir, experiment_id=None, run_id=None):
        path = Path(save_dir)/experiment_id/run_id/'artifacts'/'models'
        return path
    
