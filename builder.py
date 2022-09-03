import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.encoder import TransformerEncoder, MultiTransformerEncoder
from Model.regressionhead import AdaptRegressionHead, AdaptConvRegressionHead, MultiAdaptConvRegressionHead, SimilarAdaptRegressionHead, DensityAdaptRegressionHead
from dataset import CaraDataset, AugmentCaraDataset, MultiCaraDataset, SlideCaraDataset, DensitySlideCaraDataset

def build_model(cfg):
    return CountModel(cfg)

class CountModel(nn.Module):
    def __init__(self, cfg):
        super(CountModel, self).__init__()
        RegressionHead = cfg.Model.RegressionHead
        Encoder = cfg.Model.Encoder
        assert RegressionHead in ['AdaptRegressionHead', 'AdaptConvRegressionHead', 'MultiAdaptConvRegressionHead', 'SimilarAdaptRegressionHead', 'DensityAdaptRegressionHead'], NotImplementedError
        assert Encoder in ['TransformerEncoder', 'MultiTransformerEncoder'], NotImplementedError

        if RegressionHead == 'AdaptRegressionHead':
            self.regressionhead = AdaptRegressionHead(cfg)

        elif RegressionHead == 'AdaptConvRegressionHead':
            self.regressionhead = AdaptConvRegressionHead(cfg)

        elif RegressionHead == 'MultiAdaptConvRegressionHead':
            self.regressionhead = MultiAdaptConvRegressionHead(cfg)

        elif RegressionHead == 'SimilarAdaptRegressionHead':
            self.regressionhead = SimilarAdaptRegressionHead(cfg)

        elif RegressionHead == 'DensityAdaptRegressionHead':
            self.regressionhead = DensityAdaptRegressionHead(cfg)

        if Encoder == 'TransformerEncoder':
            self.encoder = TransformerEncoder(cfg)

        elif Encoder == 'MultiTransformerEncoder':
            self.encoder = MultiTransformerEncoder(cfg)

    def forward(self, input):
        input = self.encoder(input)
        output = self.regressionhead(input)
        return output

def build_dataset(cfg):
    datasetname = cfg.Dataset.Name
    assert datasetname in ['AugmentCaraDataset', 'CaraDataset', 'MultiCaraDataset', 'SlideCaraDataset', 'DensitySlideCaraDataset'], NotImplementedError
    if datasetname == 'AugmentCaraDataset':
        train_set = AugmentCaraDataset(cfg, data_split='Train')
        val_set = CaraDataset(cfg, data_split='Val')

    elif datasetname == 'CaraDataset':
        train_set = CaraDataset(cfg, data_split='Train')
        val_set = CaraDataset(cfg, data_split='Val')

    elif datasetname == 'MultiCaraDataset':
        train_set = MultiCaraDataset(cfg, data_split='Train')
        val_set = MultiCaraDataset(cfg, data_split='Val')

    elif datasetname == 'SlideCaraDataset':
        train_set = SlideCaraDataset(cfg, data_split='Train', root_dir='/home/yifeng/SignalCounting')
        val_set = SlideCaraDataset(cfg, data_split='Val', root_dir='/home/yifeng/SignalCounting')

    elif datasetname == 'DensitySlideCaraDataset':
        train_set = DensitySlideCaraDataset(cfg, data_split='Train', root_dir='/home/yifeng/SignalCounting')
        val_set = DensitySlideCaraDataset(cfg, data_split='Val', root_dir='/home/yifeng/SignalCounting')
    #val_set = CaraDataset(data_split='Val', root_dir='/home/yifeng/SignalCounting')

    return train_set, val_set

