import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import TransformerEncoderLayer

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FixedPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()

        self.max_len = cfg.TransformerEncoder.max_len
        self.inter_dim = cfg.TransformerEncoder.inter_dim
        self.head_num = cfg.TransformerEncoder.head_num
        self.dropout = cfg.TransformerEncoder.dropout
        self.freeze = cfg.TransformerEncoder.freeze
        self.feat_dim = cfg.Dataset.Window_size * 6
        print(self.feat_dim)
        self.activation = cfg.TransformerEncoder.activation
        self.layers_num = cfg.TransformerEncoder.layers_num

        self.project_inp = nn.Linear(self.feat_dim, self.inter_dim)
        self.pos_enc = FixedPositionalEncoding(self.inter_dim, dropout = self.dropout * (1.0 - self.freeze),
                                               max_len = self.max_len)
        encoder_layer = TransformerEncoderLayer(self.inter_dim, self.head_num, self.inter_dim,
                                                self.dropout * (1.0 - self.freeze), activation = self.activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.layers_num)
        if self.activation == 'gelu':
            self.act = F.gelu
        else:
            self.act = F.relu

    def forward(self, input):
        # B x L x D -> L x B x D
        input = input.permute(1, 0, 2)
        #print('in', input.shape)
        input = self.project_inp(input) * math.sqrt(self.inter_dim)
        #print(input.shape)
        input = self.pos_enc(input)
        output = self.transformer_encoder(input)
        output = self.act(output)
        return output

class MultiTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(MultiTransformerEncoder, self).__init__()

        self.max_len = cfg.TransformerEncoder.max_len
        self.inter_dim = cfg.TransformerEncoder.inter_dim
        self.head_num = cfg.TransformerEncoder.head_num
        self.dropout = cfg.TransformerEncoder.dropout
        self.freeze = cfg.TransformerEncoder.freeze
        self.window_size_list = cfg.MultiCaraDataset.Window_size_list

        self.activation = cfg.TransformerEncoder.activation
        self.layers_num = cfg.TransformerEncoder.layers_num

        self.project_inp_list = []
        for index in range(len(self.window_size_list)):
            self.project_inp_list.append(nn.Linear(self.window_size_list[index] * 6, self.inter_dim).cuda())

        #self.project_inp_list.append(nn.Linear(60, self.inter_dim).cuda())
        #self.project_inp_list.append(nn.Linear(90, self.inter_dim).cuda())

        #self.project_inp = nn.Linear(self.feat_dim, self.inter_dim)
        self.pos_enc = FixedPositionalEncoding(self.inter_dim, dropout = self.dropout * (1.0 - self.freeze),
                                               max_len = self.max_len)
        encoder_layer = TransformerEncoderLayer(self.inter_dim, self.head_num, self.inter_dim,
                                                self.dropout * (1.0 - self.freeze), activation = self.activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.layers_num)
        if self.activation == 'gelu':
            self.act = F.gelu
        else:
            self.act = F.relu

    def forward(self, input_list):
        # B x L x D -> L x B x D
        output_list = []
        for i in range(3):
            input = input_list[i]
            input = input.cuda()
            input = input.permute(1, 0, 2)
            input = self.project_inp_list[i](input) * math.sqrt(self.inter_dim)
            input = self.pos_enc(input)
            output = self.transformer_encoder(input)
            output = self.act(output)
            output_list.append(output)
        return output_list