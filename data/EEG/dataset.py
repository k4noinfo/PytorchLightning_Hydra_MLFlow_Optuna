import torch
import pathlib
import pandas as pd
import numpy as np

from scipy.io import arff

import omegaconf

from sklearn.preprocessing import MinMaxScaler

from src.utils.dataset import TSDatasetBase

class TSDataset(torch.utils.data.Dataset, TSDatasetBase):
    def __init__(self, config: omegaconf.dictconfig.DictConfig, 
                 mode:str, data_path:pathlib.Path, label_file=None):
        """
        Args:
            config:
            mode:
            data_path
        """
        super(self.__class__, self).__init__(config, mode, data_path, label_file=None)
        
        self._x = None
        self._y = None
        
        scaler = MinMaxScaler(feature_range=(0.01,0.99))
        
        for file_path in self.data_file_paths:
            
            _x = None
            
            if file_path.suffix == '.csv':
                _x = pd.read_csv(file_path, index_col=config.time_idx, parse_dates=True).values
            elif file_path.suffix == '.arff':
                _x, _meta = arff.loadarff(str(file_path))
                _x = pd.DataFrame(_x)
                _x_value = _x.values[:,[i for i in range(_x.shape[1]-1)]]
                _x_label = _x.values[:,[-1]]
                _x_value = scaler.fit_transform(_x_value)
                _x = pd.DataFrame(np.hstack([_x_value, _x_label]), columns=_x.columns)
                if mode in ['train','val']:
                    _x = _x[_x['eyeDetection']==b'1'] # eye close
                if 'time_idx' in config:
                    if config.time_idx is not None:
                        _x.set_index(config.time_idx)
            _x = _x.values
            
            if 'feature_cols' in config:
                if not self.feature_cols: #empty check
                    self.feature_cols = [i for i in range(_x.shape[1])]
                _x = _x[:,self.feature_cols]
            
            
            if 'split_size' in eval(f'config.{mode}'):
                if mode == 'train':
                    _x = _x[:int(_x.shape[0]*config.train.split_size)]
                elif mode == 'val':
                    _x = _x[int(_x.shape[0]*(1-config.val.split_size)):]
                    
            n_time, n_features = _x.shape
            data = []
            # split data for each time window size and slide window
            for i in range(0, n_time - self.window_size+1, self.slide_step):
                if 'input_vec' in config:
                    if config.input_vec == 'time':
                        data.append(_x[i:i+self.window_size].T)
                    elif config.input_vec == 'features':
                        data.append(_x[i:i+self.window_size])
                    else:
                        raise f"error {config.input_vec}"
                else:
                    data.append(_x[i:i+self.window_size])

            data = np.array(data, dtype='float64') # 要再検討
            
            label = []
            if self.target_type == 'predict': # 要動作確認
                for ts in data[1:]:
                    label.append(ts[self.pred_interval][self.target_cols])
                label = np.array(label, dtype='float64') # 要再検討
                label = label.astype(float)
                data = data[:-1]
                pass
            elif self.target_type == 'generative':
                label = np.array(data)
            else:
                pass
            #print(label)
            if self._x is None:
                self._x = data
            else:
                self._x = np.append(self._x, np.array(data), axis=0)
                
            if self._y is None:
                self._y = label
            else:
                self._y = np.append(self._y, label, axis=0)
            print(mode, ' (x,y): ', data.shape, label.shape)
        print(mode, 'total: (x,y): ', self._x.shape, self._y.shape)
        
        if len(self._x) != len(self._y):
            try:
                raise ValueError('length differnece between x and y.' + 
                                 str(len(self._x)) + ',' + str(len(self._y)))
            except ValueError as e:
                print(e)      

    def __getitem__(self, index):
        # TODO:　numpy 形式でほしいか、 Tensor 形式でほしいのか、config で設定してもよいかも
        #print(type(self._x[index]), self._x[index])
        #return self._x[index], self._y[index]
        # 元データのせいか、これがないと RuntimeError: expected scalar type Double but found Float がでる...
        return torch.Tensor(self._x[index]).float(), torch.Tensor(self._y[index]).float() 
        #return np.array(self._x[index], dtype='float64'), np.array(self._y[index], dtype='float64')

    def __len__(self):
        return self._x.shape[0]

'''未実装'''
class TSIterableDataset(torch.utils.data.IterableDataset, TSDatasetBase):
    def __init__(self, config:omegaconf.dictconfig.DictConfig, mode:str, data_path:pathlib.Path):
        super(self.__class__, self).__init__(config, mode, data_path)
    def __iter__(self):
        pass
    def __getitem__(self, idx):
        pass
    def __len__(self):
        pass
    