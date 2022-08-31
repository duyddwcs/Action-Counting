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

#    default_dict = {}
#
#    # Train
#    default_dict['Train'] = {}

#test_dict = {}
#test_dict['Train'] = {}
#test_dict['Train']['Epoch'] = 25
#test_dict['Train']['LR'] = 1e-5

#cfg = OmegaConf.create(test_dict)
#print(cfg)
#print(cfg.Train.Epoch)
#cfg2 = OmegaConf.load('Config/default.yaml')
#print(cfg2)

#cfg3 = OmegaConf.merge(cfg2, cfg)
#print(cfg3)