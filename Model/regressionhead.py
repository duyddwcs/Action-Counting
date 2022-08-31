import torch
import torch.nn as nn
import torch.nn.functional as F

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

        if cfg.Dataset.Density_map:
            self.outdim = cfg.Dataset.Density_map_length
        else:
            self.outdim = 1

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

'''
Add convolution layer to reduce the feature dim
'''

class AdaptConvRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(AdaptConvRegressionHead, self).__init__()

        self.pool_len = cfg.AdaptRegressionHead.pool_len
        self.mlp_dim = cfg.AdaptRegressionHead.mlp_dim
        self.act = cfg.AdaptRegressionHead.act
        self.indim = self.pool_len * cfg.AdaptConvRegressionHead.conv_out_dim

        assert self.act in ['relu', 'gelu']

        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)

        if cfg.Dataset.Density_map:
            self.outdim = cfg.Dataset.Density_map_length
        else:
            self.outdim = 1

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

        self.conv = []
        self.conv.append(torch.nn.Conv1d(cfg.TransformerEncoder.inter_dim, 128, 3, stride=1, padding=1))
        if self.act == 'relu':
            self.conv.append(nn.ReLU())
        else:
            self.conv.append(nn.GELU())
        self.conv = nn.Sequential(*self.conv)


    def forward(self, input):
        # input : L X B X C -> B x C x L
        input = input.permute(1, 2, 0)
        input = self.adapool(input)
        input = self.conv(input)
        input = torch.flatten(input)
        output = self.regressor(input)
        output = F.relu(output)
        return output

class DensityAdaptRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(DensityAdaptRegressionHead, self).__init__()

        self.pool_len = cfg.DensityAdaptRegressionHead.pool_len
        self.mlp_dim = cfg.DensityAdaptRegressionHead.mlp_dim
        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)
        self.indim = self.pool_len * cfg.TransformerEncoder.inter_dim
        self.outdim = 128

        self.regressor = []
        if len(self.mlp_dim) == 0:
            self.regressor.append(nn.Linear(self.indim, self.outdim))
            self.regressor.append(nn.ReLU())

        else:
            self.regressor.append(nn.Linear(self.indim, self.mlp_dim[0]))
            self.regressor.append(nn.ReLU())

            for i in range(len(self.mlp_dim) - 1):
                self.regressor.append(nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]))
                self.regressor.append(nn.ReLU())

            self.regressor.append(nn.Linear(self.mlp_dim[-1], self.outdim))
            self.regressor.append(nn.ReLU())

        self.regressor = nn.Sequential(*self.regressor)

    def forward(self, input):
        # input : L X B X C -> B x C x L
        input = input.permute(1, 2, 0)
        input = self.adapool(input)
        input = torch.flatten(input)
        output = self.regressor(input)
        output = F.relu(output)
        return output

'''class AdaptConvRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(AdaptConvRegressionHead, self).__init__()

        self.pool_len = cfg.AdaptConvRegressionHead.pool_len
        self.mlp_dim = cfg.AdaptConvRegressionHead.mlp_dim
        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)
        self.conv = nn.Conv1d(96, 48, 7, stride=3)
        self.indim = 48 * 96
        self.outdim = 1

        self.regressor = []
        if len(self.mlp_dim) == 0:
            self.regressor.append(nn.Linear(self.indim, self.outdim))
            self.regressor.append(nn.ReLU())

        else:
            self.regressor.append(nn.Linear(self.indim, self.mlp_dim[0]))
            self.regressor.append(nn.ReLU())

            for i in range(len(self.mlp_dim) - 1):
                self.regressor.append(nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]))
                self.regressor.append(nn.ReLU())

            self.regressor.append(nn.Linear(self.mlp_dim[-1], self.outdim))
            self.regressor.append(nn.ReLU())

        self.regressor = nn.Sequential(*self.regressor)
    def forward(self, input):
        # input : L X B X C -> B x C x L
        input = input.permute(1, 2, 0)
        input = self.conv(input)
        input = F.relu(input)
        #print(input.shape)
        input = self.adapool(input)
        #print(input.shape)

        input = torch.flatten(input)
        output = self.regressor(input)
        return output'''

'''
class MultiAdaptConvRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(MultiAdaptConvRegressionHead, self).__init__()

        self.pool_len = cfg.AdaptConvRegressionHead.pool_len
        self.mlp_dim = cfg.AdaptConvRegressionHead.mlp_dim
        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)
        self.conv1 = nn.Conv1d(96, 64, 3, stride=3)
        self.conv2= nn.Conv1d(64, 32, 3, stride=3)
        self.indim = 32 * 32
        self.outdim = 1

        self.regressor = []
        if len(self.mlp_dim) == 0:
            self.regressor.append(nn.Linear(self.indim, self.outdim))
            self.regressor.append(nn.ReLU())

        else:
            self.regressor.append(nn.Linear(self.indim, self.mlp_dim[0]))
            self.regressor.append(nn.ReLU())

            for i in range(len(self.mlp_dim) - 1):
                self.regressor.append(nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]))
                self.regressor.append(nn.ReLU())

            self.regressor.append(nn.Linear(self.mlp_dim[-1], self.outdim))
            self.regressor.append(nn.ReLU())

        self.regressor = nn.Sequential(*self.regressor)
    def forward(self, input_list):
        # input : L X B X C -> B x C x L
        final_feat = None
        for input in input_list:
            input = input.permute(1, 2, 0)
            #print(input.shape)
            # B x C x L
            input = self.adapool(input).permute(0, 2, 1)
            if final_feat is None:
                final_feat = input
            else:
                final_feat = torch.cat((final_feat, input), 2)
        #print(final_feat.shape)
        final_feat = self.conv1(final_feat)
        final_feat = F.gelu(final_feat)
        final_feat = self.conv2(final_feat)
        final_feat = F.gelu(final_feat)
        #print(final_feat.shape)
        final_feat = torch.flatten(final_feat)
        #print(final_feat.shape)
        final_feat = self.regressor(final_feat)
        #print(final_feat)
        return final_feat
'''

class MultiAdaptConvRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(MultiAdaptConvRegressionHead, self).__init__()

        self.pool_len = cfg.MultiAdaptConvRegressionHead.pool_len
        self.mlp_dim = cfg.MultiAdaptConvRegressionHead.mlp_dim
        self.act = cfg.MultiAdaptConvRegressionHead.act
        self.indim = self.pool_len * cfg.MultiAdaptConvRegressionHead.conv_out_dim[-1]

        assert self.act in ['relu', 'gelu']

        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)

        if cfg.Dataset.Density_map:
            self.outdim = cfg.Dataset.Density_map_length
        else:
            self.outdim = 1

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

        # Convolution

        self.conv = []
        conv_dim_list = cfg.MultiAdaptConvRegressionHead.conv_out_dim
        # Add the input dim
        conv_dim_list.insert(0, cfg.TransformerEncoder.inter_dim * 3)
        for index in range(len(conv_dim_list) - 1):
            self.conv.append(torch.nn.Conv1d(conv_dim_list[index], conv_dim_list[index + 1], 3, stride=1, padding=1))
            if self.act == 'relu':
                self.conv.append(nn.ReLU())
            else:
                self.conv.append(nn.GELU())
        self.conv = nn.Sequential(*self.conv)


    def forward(self, input_list):
        # input : L X B X C -> B x C x L
        final_feat = None
        for input in input_list:
            input = input.permute(1, 2, 0)
            # print(input.shape)
            # B x C x L
            input = self.adapool(input)
            if final_feat is None:
                final_feat = input
            else:
                final_feat = torch.cat((final_feat, input), 1) #Channel dim
        #print('###############')
        #print(final_feat.shape)
        input = self.conv(final_feat)
        #print(input.shape)
        input = torch.flatten(input)
        #print(input.shape)
        output = self.regressor(input)
        output = F.relu(output)
        return output


'''class SimilarAdaptRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(SimilarAdaptRegressionHead, self).__init__()

        self.pool_len = cfg.SimilarAdaptRegressionHead.pool_len
        self.mlp_dim = cfg.SimilarAdaptRegressionHead.mlp_dim
        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)
        self.indim = self.pool_len * self.pool_len
        self.outdim = 1

        self.attention_simi = SelfAttentionSimilarity(dim = cfg.TransformerEncoder.inter_dim)
        self.conv1 = nn.Conv2d(4, 4, 3, stride = 2, padding = 1)

        self.regressor = []
        if len(self.mlp_dim) == 0:
            self.regressor.append(nn.Linear(self.indim, self.outdim))
            self.regressor.append(nn.GELU())

        else:
            self.regressor.append(nn.Linear(self.indim, self.mlp_dim[0]))
            self.regressor.append(nn.GELU())

            for i in range(len(self.mlp_dim) - 1):
                self.regressor.append(nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]))
                self.regressor.append(nn.GELU())

            self.regressor.append(nn.Linear(self.mlp_dim[-1], self.outdim))
            self.regressor.append(nn.GELU())

        self.regressor = nn.Sequential(*self.regressor)
    def forward(self, input):
        #print(input.shape)
        # input : L X B X C -> B x C x L
        input = input.permute(1, 2, 0)
        input = self.adapool(input)
        # B x L x C
        #print(input.shape)
        input = input.permute(0, 2, 1)
        input = self.attention_simi(input)
        input = self.conv1(input)
        input = F.relu(input)
        # B x C x L
        #print(input.shape)
        input = torch.flatten(input)
        output = self.regressor(input)
        return output'''

class SimilarAdaptRegressionHead(nn.Module):
    def __init__(self, cfg):
        super(SimilarAdaptRegressionHead, self).__init__()

        self.pool_len = cfg.SimilarAdaptRegressionHead.pool_len
        self.mlp_dim = cfg.SimilarAdaptRegressionHead.mlp_dim
        self.act = cfg.SimilarAdaptRegressionHead.act
        self.indim = self.pool_len * self.pool_len * 4
        self.attention_simi = SelfAttentionSimilarity(dim = cfg.SimilarAdaptRegressionHead.conv_out_dim)
        assert self.act in ['relu', 'gelu']

        self.adapool = nn.AdaptiveAvgPool1d(self.pool_len)

        if cfg.Dataset.Density_map:
            self.outdim = cfg.Dataset.Density_map_length
        else:
            self.outdim = 1

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

        self.conv = []
        self.conv.append(torch.nn.Conv1d(cfg.TransformerEncoder.inter_dim, cfg.SimilarAdaptRegressionHead.conv_out_dim, 3, stride=1, padding=1))
        if self.act == 'relu':
            self.conv.append(nn.ReLU())
        else:
            self.conv.append(nn.GELU())
        self.conv = nn.Sequential(*self.conv)


    def forward(self, input):
        # input : L X B X C -> B x C x L
        input = input.permute(1, 2, 0)
        input = self.adapool(input)
        input = self.conv(input)
        input = self.attention_simi(input)
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

#model = SelfAttentionSimilarity(dim = 96)
#input = torch.rand((1, 64, 96))
#simi = model(input)
#print(simi.shape)


#64 x 96
#2) 1 d convolution -> 64 x 32
#1) 64 x 64 -> flaten regression

# input1 = torch.rand((1, 4, 128, 128))#.permute(0, 2, 1)
#input2 = torch.rand((60, 96, 96))#.permute(0, 2, 1)
#input3 = torch.rand((90, 96, 96))#.permute(0, 2, 1)
# conv1 = nn.Conv2d(4, 4, 3, stride = 2, padding = 1)
# out = conv1(input1)
# print(out.shape)
#conv2 = nn.Conv1d(96, 48, 5, stride=3)
#conv3 = nn.Conv1d(96, 48, 3, stride=3)
#out1 = conv1(input3)
#out2 = conv2(input3)
#out3 = conv3(input3)
#print(out1.shape)
#print(out2.shape)
#print(out3.shape)
#input1 = torch.rand((30, 1, 96))
#input2 = torch.rand((60, 96, 96))
#input3 = torch.rand((90, 96, 96))
#featlist = [input1, input2, input3]
#model =
'''model = AdaptConvRegressionHead()
input3 = torch.rand((2498, 1, 96))
out = model(input3)'''

'''
96 * 96
96 * 96
'''