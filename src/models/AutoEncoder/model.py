import torch.nn.functional as F

import pytorch_lightning as pl

from hydra.utils import instantiate

from omegaconf import OmegaConf

from .modules import Encoder, Decoder

class AutoEncoder(pl.LightningModule):
    def __init__(self, cfg):
        '''
        全結合ネットワークのAutoEncoder
        
        Parameters:
        --------------------------------------
        config:
            設定ファイルから読み込んだ定数リスト
            ネットワーク構造などのパラメタの設定も設定ファイルで基本行う
        
        '''
        super(AutoEncoder, self).__init__()
        
        self.config = cfg
        if 'input_shape' in cfg.data:
            self.n_features = cfg.data.input_shape[0]
            if len(cfg.data.feature_cols) != self.n_features:
                self.n_features = len(cfg.data.feature_cols)
        else:
            self.n_features = len(cfg.data.feature_cols)
            #self.config.data.input_shape = [self.n_features]
        self.n_timesteps = cfg.data.window_size
        self.hidden_size = OmegaConf.to_container(cfg.net.hidden_size)
        self.z_dim = cfg.net.z_dim
        self.dropout = cfg.net.dropout
        
        self.anomaly_scores = None
        self.recon_x = None
        
        act_f = None
        if 'act_f' in self.config.net:
            act_f = eval(self.config.net.act_f)
        
        ''' 入力を時間(time)軸にするか、特徴量(features)軸にするか '''
        if 'input_vec' in self.config.data.keys():
            if self.config.data.input_vec == 'time':
                self.encoder = Encoder(self.n_timesteps, self.hidden_size, 
                                       self.z_dim, self.n_features, self.dropout)
                self.decoder = Decoder(self.n_timesteps, self.hidden_size,
                                       self.z_dim, self.n_features, self.dropout)
            else:
                self.encoder = Encoder(self.n_features, self.hidden_size, 
                                       self.z_dim, self.n_timesteps, self.dropout)
                self.decoder = Decoder(self.n_features, self.hidden_size, 
                                       self.z_dim, self.n_timesteps, self.dropout)
        else:
            raise ValueError(f'config[\'data\'] has not \'input_vec\'')
        
        
        self.loss = F.mse_loss
        if 'loss_f' in self.config.net:
            self.loss = eval(self.config.net.loss_f) # instantiate のほうがいい？
        
        self.optimizer = instantiate(self.config.optimizer, params=self.parameters())
        
        self.save_hyperparameters(OmegaConf.to_container(self.config.net, resolve=True))

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z
    
    #def train_loss(self, x):
    # 複数の optimizer がある場合には training_step(self, batch, batch_idx, optimizer_idx) となるぽいう
    def training_step(self, batch, batch_idx):
        '''
        Parameters
        --------------------
        batch (list): (input_tensor, output_tensor)
        batch_idx:
        '''
        x, y = batch
        recon_x, _ = self(x)
        #print(recon_x.size(), x.size())
        
        #loss = F.cross_entropy(recon_x, x)
        #loss = F.mse_loss(recon_x, x)
        loss = self.loss(recon_x, x, reduction='sum')
        
        # 学習を resume したときに step が保存されていないとリセットされてしまうため
        loss_dict = {'loss': loss.item(), 'step': self.global_step}
        self.log_dict(loss_dict)
        return loss
    
        # multi-gpu とかの場合につかうようす
    # def training_step_end(self, training_step_output)
    
    # 複数のvalidation用dataloaderがある場合には、dataloader_idx も加えるぽい
    # def validation_step(self, batch, batch_idx, dataloader_idx)
    def validation_step(self, batch, batch_idx):
        '''
        Parameters
        --------------------
        batch (list): (input_tensor, output_tensor)
        batch_idx:
        '''
        x, y = batch
        recon_x, _ = self(x)
        loss = self.loss(recon_x, x, reduction='sum')
        
        loss_dict = {'val_loss':loss.item()}
        self.log_dict(loss_dict, prog_bar=True)
    
    # multi-gpu とかの場合につかうようす
    # def validation_step_end(self, training_step_output)
    
    # validation_step が実装されていないと呼ばれない。複数dataloader用?
    # def validaition_epoch_end(self, outputs):
    
    # anomaly_score の計算などを行う
    def test_step(self, batch, batch_idx):
        x, y = batch
        recon_x, _ = self(x)
        loss = self.loss(recon_x, x, reduction='sum')
        self.log_dict({'test_loss':loss.item()})
        
        if self.anomaly_scores is None:
            self.anomaly_scores = []
            
        error_vec = (recon_x - x).abs()
        if self.config.data.input_vec == 'time':
            anomaly_score = error_vec.sum(-1, keepdim=False).cpu().data.numpy()
        elif self.config.data.input_vec == 'features':
            anomaly_score = error_vec.sum(axis=1, keepdim=False).cpu().data.numpy()
        else:
            anomaly_score = error_vec.sum(-1, keepdim=False).cpu().data.numpy()
        
        anomaly_score = anomaly_score.reshape(anomaly_score.shape[1])
        
        self.anomaly_scores.append(anomaly_score)
        
        if self.recon_x is None:
            self.recon_x = []
        self.recon_x.append(recon_x.view(recon_x.size(1), recon_x.size(2)).cpu().numpy())

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=1e-3)
        return self.optimizer

##################
# for Optuna
##################
class AutoEncoder_PS(AutoEncoder):
    def __init__(self, cfg, trial):
        ''' 探索したいパラメータを登録して値を config に書き込みモデルを生成 '''
        
        # netwark configuration
        cfg.net.z_dim = trial.suggest_int('model.net.z_dim', cfg.optuna.net.z_dim.min, 
                                          cfg.optuna.net.z_dim.max)
        self.n_layers = trial.suggest_int('model.net.n_layers', cfg.optuna.net.n_layers.min,
                                          cfg.optuna.net.n_layers.max)
        #cfg.net.dropout  = trial.suggest_float('dropout', cfg.optuna.net.dropout.min, cfg.optuna.net.dropout.max)    
        cfg.net.hidden_size = []
        for i in range(self.n_layers):
            h_size = trial.suggest_int('model.net.n_units_l{}'.format(i), cfg.optuna.net.hidden_size.min, 
                                       cfg.optuna.net.hidden_size.max, log=True)
            cfg.net.hidden_size.append(h_size)
        
        # learning_rate
        cfg.optimizer.lr = trial.suggest_loguniform('model.optimizer.lr', cfg.optuna.optimizer.lr.min,
                                                    cfg.optuna.optimizer.lr.max)
        
        super(AutoEncoder_PS, self).__init__(cfg)