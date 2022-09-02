import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import TransformerEncoderLayer
from Model.encoder import FixedPositionalEncoding
import math
'''
Base regression head
'''
class AdaptRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(AdaptRegressionHead, self).__init__()

        self.pool_len = cfg.AdaptRegressionHead.pool_len
        self.mlp_dim = cfg.AdaptRegressionHead.mlp_dim
        self.act = cfg.AdaptRegressionHead.act

        assert self.act in ['relu', 'gelu']

        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)

        self.indim = self.pool_len * cfg.TransformerEncoder.inter_dim

        if cfg.CARA.Density_map and cfg.Dataset == 'CaraDataset':
            self.outdim = cfg.CARA.Density_map_length
        else:
            self.outdim = 1

        # Regression head
        self.regressor = []
        if len(self.mlp_dim) == 0:
            self.regressor.append(nn.Linear(self.indim, self.outdim))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

        else:
            self.regressor.append(nn.Linear(self.indim, self.mlp_dim[0]))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

            for i in range(len(self.mlp_dim) - 1):
                self.regressor.append(nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]))
                if self.act == 'relu':
                    self.regressor.append(nn.ReLU())
                else:
                    self.regressor.append(nn.GELU())

            self.regressor.append(nn.Linear(self.mlp_dim[-1], self.outdim))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

        self.regressor = nn.Sequential(*self.regressor)

    def forward(self, input):
        # input : L X B X C -> B x C x L
        input = input.permute(1, 2, 0)
        input = self.adapool(input)
        input = torch.flatten(input)
        output = self.regressor(input)
        output = F.relu(output)
        return output

class TSMRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(TSMRegressionHead, self).__init__()
        self.pool_len = cfg.TSMRegressionHead.pool_len
        self.mlp_dim = cfg.TSMRegressionHead.mlp_dim
        self.act = cfg.TSMRegressionHead.act

        self.temperature = cfg.TSMRegressionHead.temperature

        if cfg.CARA.Density_map and cfg.Dataset == 'CaraDataset':
            self.outdim = cfg.CARA.Density_map_length
        else:
            self.outdim = 1

        assert self.act in ['relu', 'gelu']
        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)

        # Convolution layer
        self.conv = nn.Conv2d(in_channels=1, out_channels = cfg.TSMRegressionHead.conv_out_dim, kernel_size=3, padding=1)

        # Transformer
        self.project_inp = nn.Linear(cfg.TSMRegressionHead.conv_out_dim * self.pool_len, cfg.TSMRegressionHead.transformer_dim)
        self.pos_enc = FixedPositionalEncoding(cfg.TSMRegressionHead.transformer_dim, dropout=0.1, max_len=self.pool_len)
        encoder_layer = TransformerEncoderLayer(cfg.TSMRegressionHead.transformer_dim, cfg.TSMRegressionHead.transformer_head, cfg.TSMRegressionHead.transformer_dim, 0.1, activation=self.act)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.TSMRegressionHead.transformer_layer_num)
        self.indim = self.pool_len * cfg.TSMRegressionHead.transformer_dim

        # Regression
        self.regressor = []
        if len(self.mlp_dim) == 0:
            self.regressor.append(nn.Linear(self.indim, self.outdim))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

        else:
            self.regressor.append(nn.Linear(self.indim, self.mlp_dim[0]))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

            for i in range(len(self.mlp_dim) - 1):
                self.regressor.append(nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]))
                if self.act == 'relu':
                    self.regressor.append(nn.ReLU())
                else:
                    self.regressor.append(nn.GELU())

            self.regressor.append(nn.Linear(self.mlp_dim[-1], self.outdim))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

        self.regressor = nn.Sequential(*self.regressor)

    def get_sims(self, embs, temperature):

        batch_size = embs.shape[0]
        seq_len = embs.shape[2]
        embs = torch.reshape(embs, [batch_size, -1, seq_len])

        def _get_sims(embs: torch.Tensor):

            dist = self.pairwise_l2_distance(embs, embs)
            sims = -1.0 * dist
            return sims

        sims = self.map_fn(_get_sims, embs)
        # sims = torch.Size[20, 64, 64]
        sims /= temperature
        sims = F.softmax(sims, dim=-1)
        sims = sims.unsqueeze(dim=-1)
        return sims

    def map_fn(self, fn, elems):
        sims_list = []
        for i in range(elems.shape[0]):
            sims_list.append(fn(elems[i]))
        sims = torch.stack(sims_list)
        return sims

    def pairwise_l2_distance(self, a, b):

        norm_a = torch.sum(torch.square(a), dim=0)
        norm_a = torch.reshape(norm_a, [-1, 1])
        norm_b = torch.sum(torch.square(b), dim=0)
        norm_b = torch.reshape(norm_b, [1, -1])
        a = torch.transpose(a, 0, 1)
        zero_tensor = torch.zeros(self.pool_len, self.pool_len).cuda()
        dist = torch.maximum(norm_a - 2.0 * torch.matmul(a, b) + norm_b, zero_tensor)
        return dist

    def forward(self, input):
        # input : L X B X C -> B x C x L
        input = input.permute(1, 2, 0)
        input = self.adapool(input)
        input = self.get_sims(input, self.temperature)
        #print(input.shape)
        input = input.permute(0, 3, 1, 2)
        input = F.relu(self.conv(input)).squeeze()
        input = input.unsqueeze(0)
        # B L C
        input = torch.reshape(input, [1, self.pool_len, -1])
        input = input.permute(1, 0, 2)
        input = self.project_inp(input) * math.sqrt(512)
        # print(input.shape)
        input = self.pos_enc(input)
        input = self.transformer_encoder(input)
        input = F.relu(input)
        input = torch.flatten(input)
        output = self.regressor(input)
        output = F.relu(output)
        return output

class TransRACRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(TransRACRegressionHead, self).__init__()

        self.pool_len = cfg.TransRACRegressionHead.pool_len
        self.mlp_dim = cfg.TransRACRegressionHead.mlp_dim
        self.act = cfg.TransRACRegressionHead.act

        self.attention_simi = SelfAttentionSimilarity(dim = cfg.TransformerEncoder.inter_dim)
        assert self.act in ['relu', 'gelu']

        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)

        if cfg.CARA.Density_map and cfg.Dataset == 'CaraDataset':
            self.outdim = cfg.CARA.Density_map_length
        else:
            self.outdim = 1

        self.conv = []
        self.conv.append(nn.Conv2d(12, cfg.TransRACRegressionHead.conv_out_dim, 3, stride=1, padding=1))
        if self.act == 'relu':
            self.conv.append(nn.ReLU())
        else:
            self.conv.append(nn.GELU())
        self.conv = nn.Sequential(*self.conv)

        # Transformer
        self.project_inp = nn.Linear(cfg.TransRACRegressionHead.conv_out_dim * self.pool_len,
                                     cfg.TransRACRegressionHead.transformer_dim)
        self.pos_enc = FixedPositionalEncoding(cfg.TransRACRegressionHead.transformer_dim, dropout=0.1,
                                               max_len=self.pool_len)
        encoder_layer = TransformerEncoderLayer(cfg.TransRACRegressionHead.transformer_dim,
                                                cfg.TransRACRegressionHead.transformer_head,
                                                cfg.TransRACRegressionHead.transformer_dim, 0.1, activation=self.act)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.TransRACRegressionHead.transformer_layer_num)
        self.indim = self.pool_len * cfg.TransRACRegressionHead.transformer_dim

        self.regressor = []
        if len(self.mlp_dim) == 0:
            self.regressor.append(nn.Linear(self.indim, self.outdim))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

        else:
            self.regressor.append(nn.Linear(self.indim, self.mlp_dim[0]))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

            for i in range(len(self.mlp_dim) - 1):
                self.regressor.append(nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]))
                if self.act == 'relu':
                    self.regressor.append(nn.ReLU())
                else:
                    self.regressor.append(nn.GELU())

            self.regressor.append(nn.Linear(self.mlp_dim[-1], self.outdim))
            if self.act == 'relu':
                self.regressor.append(nn.ReLU())
            else:
                self.regressor.append(nn.GELU())

        self.regressor = nn.Sequential(*self.regressor)

    def forward(self, input_list):
        # Input is a list
        # input : L X B X C -> B x C x L
        multi_input = None
        for input in input_list:
            input = input.permute(1, 2, 0)
            input = self.adapool(input)
            # B x L x C
            input = input.permute(0, 2, 1)
            input = self.attention_simi(input).squeeze()
            if multi_input is None:
                multi_input = input
            else:
                multi_input = torch.cat((multi_input, input), dim=0)
        #print(multi_input.shape)

        multi_input = multi_input.unsqueeze(0)
        input = self.conv(multi_input)

        # Add a transformer
        input = torch.reshape(input, [1, self.pool_len, -1])
        input = input.permute(1, 0, 2)
        input = self.project_inp(input) * math.sqrt(512)
        input = self.pos_enc(input)
        input = self.transformer_encoder(input)
        input = F.relu(input)

        input = input.squeeze()
        input = torch.flatten(input)
        output = self.regressor(input)
        output = F.relu(output)
        return output

class SelfAttentionSimilarity(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print(x.shape)
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_similarity = self.attn_drop(attn)

        return attn_similarity


