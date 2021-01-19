import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_dim, n_features, dropout, act_f=nn.ReLU):
        '''
        多層パーセプトロン Encoder
        潜在変数は次元z_dimの多変量変数
        
        Parameters
        -------------------------------------------
        input_size: int
        
        hidden_size: int or list(int)
            隠れ層のノード数と層数を設定
        
        TODO: 入力の次元の検討 
        　1. n_timesteps にするか、
          2. n_features にするのか。
          3. n_timesteps * n_features にするのか。
        TODO:　グラフの構成の検討
        　1, 2 の場合、1つのネットワークのみか、n_features or n_timesteps 分のモデルを構築して学習するのか。
        　ただ、2だとすると、食わせるデータをスライディングウィンドウで分ける必要性はないってことか。 
        TODO:　入力の順序の検討
        　3 の場合に入力データの並べ方を考えないとだめかも
         
         とりあえず入力次元は n_timesteps にして、n_feature を列に並べて食わせるモデルにて値の変化を学習させることにする。
         中間表現は単なる多次元データとしておく
        '''
        super(self.__class__, self).__init__()
        #self.n_features = n_features
        
        self.act = act_f # config に入れたいな
        if self.act is None:
            self.act = nn.ReLU
        
        if type(hidden_size) is list:
            nodes = [input_size, *hidden_size]
        else:
            nodes = [input_size, hidden_size]
        layers = []
        
        for i in range(1,len(nodes)):
            layers.append(nn.Linear(nodes[i-1], nodes[i]))
            layers.append(self.act())
            layers.append(nn.Dropout(dropout))
        #linear_layers = [nn.Linear(nodes[i-1], nodes[i]) for i in range(1,len(nodes))]

        self.hidden = nn.ModuleList(layers)
        self.z     = nn.Linear(hidden_size[-1], z_dim)     
        
    def forward(self, x):
        if type(self.hidden) is nn.ModuleList:
            for layer in self.hidden:
                #x = self.act(layer(x))
                x = layer(x)
        else:
            x = self.act(self.hidden(x))
            
        return self.z(x)
    
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, z_dim, n_features, dropout, act_f=nn.ReLU):
        '''
        基本的にEncoderと同じ
        TODO: これも、Encoderと同じ
        '''
        super(self.__class__, self).__init__()
        
        self.act = act_f
        if self.act is None:
            self.act = nn.ReLU
            
        if type(hidden_size) is list:
            
            hidden_size.reverse()
            nodes = [z_dim, *hidden_size]
            layers = []
            for i in range(1,len(nodes)):
                layers.append(nn.Linear(nodes[i-1], nodes[i]))
                layers.append(self.act())
                layers.append(nn.Dropout(dropout))
            #linear_layers = [nn.Linear(nodes[i-1], nodes[i]) for i in range(1,len(nodes))]
            
            #self.hidden = nn.ModuleList(linear_layers)
            self.hidden = nn.ModuleList(layers)
            self.output = nn.Linear(hidden_size[-1], output_size)
        else:
            self.hidden = nn.Linear(z_dim, hidden_size)
            self.output = nn.Linear(hidden_size, output_size)
            
    def forward(self, z):
        if type(self.hidden) is nn.ModuleList:
            for layer in self.hidden:
                #z = self.act(layer(z))
                z = layer(z)
        else:
            z = self.act(self.hidden(z))
        return self.output(z)
        
    