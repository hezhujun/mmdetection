import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead


@HEADS.register_module()
class RelationNetworkBBoxHeadEncoder(BBoxHead):

    def __init__(self,
                 fc1_dim=1024,
                 fc2_dim=1024,
                 r1=1, r2=1,
                 dk=64, dg=64, Nr=16,
                 *args,
                 **kwargs):
        super(RelationNetworkBBoxHeadEncoder, self).__init__(*args, **kwargs)

        _in_channels = self.in_channels * self.roi_feat_area
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.fc1 = nn.Linear(_in_channels, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.r1 = r1
        self.r2 = r2
        self.dk = dk
        self.dg = dg
        self.Nr = Nr
        if r1 > 0:
            encoder_layer1 = nn.TransformerEncoderLayer(d_model=fc1_dim, nhead=Nr, dim_feedforward=dk)
            self.rm1 = nn.TransformerEncoder(encoder_layer1, num_layers=r1)
        if r2 > 0:
            encoder_layer2 = nn.TransformerEncoderLayer(d_model=fc2_dim, nhead=Nr, dim_feedforward=dk)
            self.rm2 = nn.TransformerEncoder(encoder_layer2, num_layers=r2)

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.fc2_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = nn.Linear(self.fc2_dim, out_dim_reg)

    def init_weights(self):
        super(RelationNetworkBBoxHeadEncoder, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.fc1, self.fc2]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, rois):
        """

        :param x: (BS, num_rois, C, roi_h, roi_w)
        :param rois: (BS, num_rois, 5)
        :return:
        """
        BS, num_rois, C, H, W = x.shape  # BS, num_rois, C, roi_h, roi_w

        x = x.flatten(0, 1)  # BS*num_rois, C, roi_h, roi_w
        x = x.flatten(1)  # BS*num_rois, num_features
        x = F.relu(self.fc1(x))

        N, E = x.shape
        if self.r1 > 0:
            x = self.rm1(x.reshape(N, 1, E))
            x = x.reshape(N, E)

        x = F.relu(self.fc2(x))

        if self.r2 > 0:
            x = self.rm2(x.reshape(N, 1, E))
            x = x.reshape(N, E)

        x_cls = x
        x_reg = x

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
