import torch
import torch.nn as nn
import timm
from torch import nn
from einops import rearrange
from functools import partial
from timm.models.layers.helpers import to_2tuple
from timm.models.layers.drop import DropPath


def creat_model(config, model_weight_path=None, pretrained=True):
    model_oiqa = VSM_OIQA(embed_dim=config.embed_dim, seq_layers=config.seq_layers, conv_input_dim=config.conv_input_dim,
        viewport_nums=config.viewport_nums, seq_dim=config.seq_dim, seq_hidden_dim=config.seq_hidden_dim
    )
    if pretrained:
        model_oiqa.load_state_dict(torch.load(model_weight_path), strict=False)
    return model_oiqa


class ResidualBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.act_layer = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.act_layer(self.conv1(x))
        x = self.act_layer(self.conv2(x))
        x = self.act_layer(self.conv3(x)) + shortcut
        return x


class ConvLayer(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.shallow_conv = nn.Conv2d(input_dim, embed_dim, 3, 1, 1)
        self.downsample_first = nn.Conv2d(embed_dim, embed_dim, 4, 4, 0)

        self.residual_block1 = ResidualBlock(embed_dim=embed_dim)
        self.downsample1 = nn.Conv2d(embed_dim, embed_dim * 2, 3, 2, 1)

        self.residual_block2 = ResidualBlock(embed_dim=embed_dim * 2)
        self.downsample2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, 3, 2, 1)

        self.residual_block3 = ResidualBlock(embed_dim=embed_dim * 4)
        self.downsample3 = nn.Conv2d(embed_dim * 4, embed_dim * 8, 3, 2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.shallow_conv(x)
        x = self.downsample_first(x)
        x = self.downsample1(self.residual_block1(x))
        x = self.downsample2(self.residual_block2(x))
        x = self.downsample3(self.residual_block3(x))
        x = self.avgpool(x)
        return x


class SeqQA(nn.Module):
    def __init__(self, embed_dim=1024, input_dim=256, hidden_dim=256, num_layers=3, viewport_nums=6):
        super().__init__()
        self.fc_conv = nn.Linear(embed_dim, embed_dim)
        self.gelu = nn.GELU()
        self.fc_in = nn.Linear(embed_dim, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc_viewport = nn.Linear(hidden_dim, 1)
        self.fc_score = nn.Linear(viewport_nums, 1)

    def forward(self, b_x_swin, b_x_conv):
        B, M, N, C = b_x_swin.shape
        B_scores = torch.tensor([]).cpu()
        for i in range(B):
            x_swin = b_x_swin[i]
            x_conv = b_x_conv[i]

            x = x_swin + self.gelu(self.fc_conv(x_conv))
            x = self.fc_in(x)
            x, (hn, cn) = self.lstm(x)
            x = self.fc_viewport(x).flatten(1)
            x = self.fc_score(x)
            x = torch.mean(x).cpu()
            B_scores = torch.cat((B_scores, x.unsqueeze(0)), dim=0)
        B_scores = B_scores.cuda()
        return B_scores


class VSM_OIQA(nn.Module):
    def __init__(self, embed_dim=1024, seq_layers=3, conv_input_dim=128, viewport_nums=6,
            seq_dim=256, seq_hidden_dim=256):
        super().__init__()
        self.hl_extractor = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.ll_extractor = ConvLayer(input_dim=3, embed_dim=conv_input_dim)
        
        self.seq_qa = SeqQA(embed_dim=embed_dim, input_dim=seq_dim, hidden_dim=seq_hidden_dim,
            num_layers=seq_layers, viewport_nums=viewport_nums)


    def forward(self, x):
        B, M, N, C, H, W = x.shape

        # sequence features
        B_x_swin_sequence = torch.tensor([]).cpu()
        B_x_conv_sequence = torch.tensor([]).cpu()
        for i in range(B):
            x_swin_sequence = torch.tensor([]).cpu()
            x_conv_sequence = torch.tensor([]).cpu()
            for j in range(M):
                feat_swin = self.hl_extractor(x[i][j])
                feat_swin = rearrange(feat_swin, 'b (h w) c -> b c h w', h=7, w=7)
                feat_swin = self.avgpool2d(feat_swin).flatten(1).unsqueeze(0).cpu()
                feat_conv = self.ll_extractor(x[i][j]).flatten(1).unsqueeze(0).cpu()
                x_swin_sequence = torch.cat((x_swin_sequence, feat_swin), dim=0)
                x_conv_sequence = torch.cat((x_conv_sequence, feat_conv), dim=0)
            B_x_swin_sequence = torch.cat((B_x_swin_sequence, x_swin_sequence.unsqueeze(0)), dim=0)
            B_x_conv_sequence = torch.cat((B_x_conv_sequence, x_conv_sequence.unsqueeze(0)), dim=0)
        
        # B, M, N, C
        x_swin_seq = B_x_swin_sequence.cuda()
        x_conv_seq = B_x_conv_sequence.cuda()

        x = self.seq_qa(x_swin_seq, x_conv_seq)

        return x
    