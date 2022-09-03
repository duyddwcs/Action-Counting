from omegaconf import OmegaConf

def get_default_cfg():
    cfg = OmegaConf.load('./Config/default.yaml')
    return cfg

def merge_from_file(cfg, file):
    cfg2 = OmegaConf.load(file)
    cfg_new = OmegaConf.merge(cfg, cfg2)
    return cfg_new

def show_cfg(cfg):
    print(OmegaConf.to_yaml(cfg))

def save_cfg(cfg, save_path):
    OmegaConf.save(config = cfg, f = save_path)
