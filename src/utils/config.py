import yaml
import hydra

from omegaconf import OmegaConf
from hydra.experimental import initialize, compose

def load_config(config_path:str='config', config_file:str='config', overrides:dict=None):
    """
    Args:
        config_path (str):
        config_file (str):
    Return:
        config (dict):
    """
    try:
        initialize(f'../../{config_path}')
    except ValueError:
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        initialize(f'../../{config_path}')

    if overrides is not None:
        cfg = compose(config_name=config_file, overrides=overrides)
    else:
        cfg = compose(config_name=config_file)
        
    return cfg

def load_config_from_yaml(config_file):
    """
    Args:
        config_file (str):
    Return:
        config (dict):
    """
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
