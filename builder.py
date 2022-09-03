import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.encoder import TransformerEncoder, MultiTransformerEncoder
from Model.regressionhead import AdaptRegressionHead, TSMRegressionHead, TransRACRegressionHead
from dataset import MultiCaraDataset, MultiMMfit, MultiRecofit, MultiCrossfit, MultiWeakcounter

def build_model(cfg):
    return BaselineModel(cfg)

class BaselineModel(nn.Module):
    def __init__(self, cfg):
        super(BaselineModel, self).__init__()

        self.Encoder = None
        self.RegressionHead = None
        if cfg.Model == 'BasicTransformer':
            self.Encoder = TransformerEncoder(cfg)
            self.RegressionHead = AdaptRegressionHead(cfg)

        if cfg.Model == 'RepNet':
            self.Encoder = TransformerEncoder(cfg)
            self.RegressionHead = TSMRegressionHead(cfg)

        if cfg.Model == 'TransRAC':
            self.Encoder = TransformerEncoder(cfg)
            self.RegressionHead = TransRACRegressionHead(cfg)

    def forward(self, input):
        input = self.Encoder(input)
        output = self.RegressionHead(input)
        return output

def build_dataset(cfg):
    datasetname = cfg.Dataset
    assert datasetname in ['CaraDataset', 'MultiCaraDataset', 'MMFitDataset', 'RecofitDataset', 'CrossfitDataset', 'WeakcounterDataset'], NotImplementedError
    train_set = None
    val_set = None
    if datasetname == 'CaraDataset' or datasetname == 'MultiCaraDataset':
        train_set = MultiCaraDataset(cfg, data_split='Train')
        val_set = MultiCaraDataset(cfg, data_split='Val')

    if datasetname == 'MMFitDataset':
        train_set = MultiMMfit(cfg, data_split='Train')
        val_set = MultiMMfit(cfg, data_split='Val')

    if datasetname == 'RecofitDataset':
        train_set = MultiRecofit(cfg, data_split='Train')
        val_set = MultiRecofit(cfg, data_split='Val')

    if datasetname == 'CrossfitDataset':
        train_set = MultiCrossfit(cfg, data_split='Train')
        val_set = MultiCrossfit(cfg, data_split='Val')

    if datasetname == 'WeakcounterDataset':
        train_set = MultiWeakcounter(cfg, data_split='Train')
        val_set = MultiWeakcounter(cfg, data_split='Val')

    return train_set, val_set

