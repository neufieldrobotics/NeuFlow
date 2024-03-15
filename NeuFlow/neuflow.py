import torch
import torch.nn.functional as F

from NeuFlow import backbone
from NeuFlow import transformer
from NeuFlow import matching
from NeuFlow import refine
from NeuFlow import upsample
from NeuFlow import utils

from NeuFlow import config


class NeuFlow(torch.nn.Module):
    def __init__(self):
        super(NeuFlow, self).__init__()

        self.backbone = backbone.CNNEncoder(config.feature_dim)
        self.cross_attn_s16 = transformer.FeatureAttention(config.feature_dim+2, num_layers=2, bidir=True, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s16 = matching.Matching()

        self.flow_attn_s16 = transformer.FlowAttention(config.feature_dim+2)

        self.merge_s8 = torch.nn.Sequential(torch.nn.Conv2d((config.feature_dim+2) * 2, config.feature_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(config.feature_dim * 2, config.feature_dim, kernel_size=3, stride=1, padding=1, bias=False))

        self.refine_s8 = refine.Refine(config.feature_dim, patch_size=7, num_layers=6)

        self.conv_s8 = backbone.ConvBlock(3, config.feature_dim, kernel_size=8, stride=8, padding=0)

        self.upsample_s1 = upsample.UpSample(config.feature_dim, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_bhw(self, batch_size, height, width):
        self.backbone.init_pos_12(batch_size, height//8, width//8)
        self.matching_s16.init_grid(batch_size, height//16, width//16)
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, img0, img1):

        flow_list = []

        img0 = utils.normalize_img(img0, self.img_mean, self.img_std)
        img1 = utils.normalize_img(img1, self.img_mean, self.img_std)

        feature0_s8, feature0_s16 = self.backbone(img0)
        feature1_s8, feature1_s16 = self.backbone(img1)

        feature0_s16, feature1_s16 = self.cross_attn_s16(feature0_s16, feature1_s16)
        flow0 = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)

        flow0 = self.flow_attn_s16(feature0_s16, flow0)

        feature0_s16 = F.interpolate(feature0_s16, scale_factor=2, mode='nearest')
        feature1_s16 = F.interpolate(feature1_s16, scale_factor=2, mode='nearest')

        feature0_s8 = self.merge_s8(torch.cat([feature0_s8, feature0_s16], dim=1))
        feature1_s8 = self.merge_s8(torch.cat([feature1_s8, feature1_s16], dim=1))

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        delta_flow = self.refine_s8(feature0_s8, utils.flow_warp(feature1_s8, flow0), flow0)
        flow0 = flow0 + delta_flow

        if self.training:
            up_flow0 = F.interpolate(flow0, scale_factor=8, mode='bilinear', align_corners=True) * 8
            flow_list.append(up_flow0)

        feature0_s8 = self.conv_s8(img0)

        flow0 = self.upsample_s1(feature0_s8, flow0)
        flow_list.append(flow0)

        return flow_list
