import platform
from typing import Optional
from pathlib import Path

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import omegaconf

class DataModule(pl.LightningDataModule):
    """
    train, val, test splits and Transforms
    target dataset is time series data
    Example
    """
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig, *args, **kwargs):
        """
        Args
            cfg: data configure 
                example yaml decription
                    data:
                        name: <dataset name>
                        dataset_class: Dataset Class Name
                        num_worker: how many workers to use for loading data
                        file_type: {file|dir}
                        file_prefix: hoge # for file_type is dir
                        file_extension: .csv # for file_type is dir                        
                        time_col: time-index col id
                        feature_cols: use col id list for analysis
                        window_size: time window size
                        train:
                            batch_size:
                            slide_step: time window slide size
                            file: [] # for file
                        val:
                            ...
                        test:
                            ...
            *args:
            **kwargs:
        """
        super(DataModule,self).__init__(*args, **kwargs)
        
        if platform.system() == 'Windows' and 'num_works' in cfg:
            cfg.num_works = 0
            
        self.cfg = cfg
        self.data_name = cfg.name
        self.ds_class  = cfg.dataset_class
        self.num_workers = cfg.num_workers
    
    
    def setup(self, stage: Optional[str]=None):
        if stage == 'fit' or stage is None:
             # for train
            self.dataset_train = self._make_dataset('train')
            # for val
            self.dataset_val   = self._make_dataset('val')
        if stage == 'test' or stage is None:
            # for test
            self.dataset_test  = self._make_dataset('test')
    
    def _make_dataset(self, mode):
        
        exec(f'from data.{self.data_name}.dataset import {self.ds_class}; dataset_class={self.ds_class}')
        
        label_file = None
        if 'label' in eval(f'self.cfg.{mode}'):
            label_file = eval(f'self.cfg.{mode}.labels')
        data_path = Path('data/'+self.data_name+'/'+mode)
        if not data_path.exists():
            data_path = data_path.parent
        return eval(self.ds_class)(self.cfg, mode, data_path, label_file=label_file)
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.cfg.train.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.cfg.val.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.cfg.test.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)

######################################################################
# DatasetBase class
#  保存してある Dataset の先が、ファイルで読むのかフォルダで読むのかを指示
#  DB から直接取ってくるような場合には同数るといいのか要検討
######################################################################
class TSDatasetBase(object):
    
    def __init__(self, config, mode, data_path, label_file):
        
        assert mode in ['train', 'val', 'test', 'predict']
        
        self.config = config
        
        self.target_type = config.target_type
        assert self.target_type in ['generative','predict']
    
        if self.target_type in ['predict']: # list 前提かな？
            self.target_cols = config.input_cols
            self.pred_time   = config.pred_time
        
        self.window_size  = config.window_size
        self.slide_step   = eval(f'config.{mode}.slide_step')
        
        self.featrue_cols = []
        if 'feature_cols' in config:
            if OmegaConf.is_list(config.feature_cols):
                self.feature_cols = OmegaConf.to_container(config.feature_cols)
        
        self.data_file_type = config.file_type
        
        if not data_path.exists():
            raise FileNotFoundError(f'{data_path} does not found.')
        
        ''' yaml ファイルで train, val ファイルを直接指定することにする '''
        if self.data_file_type == 'file':
            datafiles = eval(f'config.{mode}.file')
            if OmegaConf.is_list(datafiles):
                datafiles = OmegaConf.to_container(datafiles)
            if type(datafiles) is list:
                data_path = [(data_path / file) for file in datafiles]
                self.data_file_paths = [file.resolve() for file in data_path]               
            else:
                data_path_tmp = data_path / datafiles
                if not data_path_tmp.exists():
                    data_path = data_path.parent / datafiles
                else:
                    data_path = data_path_tmp
                self.data_file_paths = [data_path.resolve()]
        elif self.data_file_type == 'dir':
            prefix = ''
            extension = 'csv'
            if 'file_prefix' in config:
                prefix    = config.file_prefix
            if 'file_extension' in config:
                extension = config.file_extension 
            data_file_paths = data_path.glob(prefix + '*.' + extension)
            self.data_file_paths = [file.resolve() for file in data_file_paths]
            
        """ label file の処理 """
        if label_file is not None:
            self.have_label = True
            self.labels = self.load_label(label_file)
        else:
            self.have_label = False
            