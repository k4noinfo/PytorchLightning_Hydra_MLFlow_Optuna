from pathlib import Path

import torch
import pytorch_lightning as pl

from hydra.utils import instantiate

from src.utils.callbacks import MyProgressBar

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
        self.modelcheckpoint = None
        
        self.init_model()
        
        # logger instance の生成
        self.logger = instantiate(cfg.logger)
        # jupyter だと validation の progressbar で無駄な改行が発生を削るため
        self.progressbar = MyProgressBar()
        
        # trainer 用 instance の生成
        self.early_stopping = None
        if self.config.trainer.use_early_stopping:
            self.early_stopping = instantiate(cfg.callbacks.EarlyStopping)
        
        model_log_dir = str(self._get_model_path(self.logger.save_dir, self.logger.experiment_id, self.logger.run_id))
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
            self.config.trainer.max_epochs = max_epochs
        
        if torch.cuda.is_available() and self.config.trainer.use_gpu:
            self.config.trainer.args.gpus = 1
        else:
            self.config.trainer.args.gpus = 0
        
        if self.config.use_earlystopping is not None:
            self.trainer = pl.Trainer(logger=self.logger,
                                      callbacks=[self.early_stopping, self.model_checkpoint, self.progressbar],
                                      **(self.config.trainer.args))
        else:
            self.trainer = pl.Trainer(logger=self.logger,
                                      callbacks=[self.model_checkpoint, self.progressbar], 
                                      **(self.config.trainer.args))
        
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
        if logger is None:
            logger = self.logger
         
        if not (run_id is None):
            folder_path = self._get_model_path(save_dir, experiment_id, run_id)
        else:
            folder_path = self._get_model_path(logger.save_dir, logger.experiment_id, logger.run_id)
        
        if n_epoch == 'last':
            if (folder_path/'last.ckpt').exists():
                checkpoint_file = 'last.ckpt'
            else:
                n_epoch = self.get_last_step(folder_path)
                checkpoint_file = f'model-epoch={n_epoch}.ckpt'
        else:
            checkpoint_file = f'model-epoch={n_epoch}.ckpt'

        checkpoint_file = str(folder_path/checkpoint_file)
        
        if run_id is None and self.model_checkpoint.best_model_path is not None:
            checkpoint_file = self.model_checkpoint.best_model_path
        
        if self.early_stopping is None:
            self.trainer = pl.Trainer(resume_from_checkpoint=checkpoint_file, logger=logger, 
                                      callbacks=[self.model_checkpoint, self.progressbar],
                                      **(self.config.trainer.args))
        else:
            self.trainer = pl.Trainer(resume_from_checkpoint=checkpoint_file, logger=logger, 
                                      callbacks=[self.early_stopping, self.model_checkpoint, self.progressbar],
                                      **(self.config.trainer.args))
    
    def train_from(self, n_epoch, max_epochs=None, 
                   logger=None, 
                   save_dir=None, experiment_id=None, run_id=None,
                   train_dataloader=None, val_dataloader=None, dm=None):
        
        if max_epochs is not None:
            self.config.trainer.max_epochs = max_epochs
        
        self.load_ckpt(n_epoch, logger, save_dir, experiment_id, run_id)
        
        if dm is not None:
            self.trainer.fit(self.model, dm)
        else:
            self.trainer.fit(self.model, train_dataloader, val_dataloader)
        #metric, best_metric = self.load_weight(step)
        #self.train(train_dataloader, val_dataloader, step=step, num_epochs=num_epochs, best_metric=best_metric)

    def val(self, val_dataloader):
        """Compute loss and metrics on val set"""
        pass

    def test(self, test_dataloader=None, dm=None):
        """Compute loss and metrics on test set"""
        # Trainer を resume したとき、なぜか test も resume して行おうとするため、以下の処理を追加... なんでこんな仕様？
        if self.trainer.resume_from_checkpoint is not None:
            if self.model_checkpoint.best_model_path != self.trainer.resume_from_checkpoint:
                self.trainer.resume_best_checkpoint = self.model_checkpoint.best_model_path
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
        if self.trainer.resume_from_checkpoint is not None:
            if self.model_checkpoint.best_model_path != self.trainer.resume_from_checkpoint:
                self.trainer.resume_from_checkpoint = self.model_checkpoint.best_model_path
                
        if self.model.anomaly_scores is None:
            self.test(dataloader, dm)
        return self.model.anomaly_scores
    
    def generate(self, dataloader=None, dm=None):
        if self.model.recon_x is None:
            self.test(dataloader, dm)
        return self.model.recon_x
                    
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