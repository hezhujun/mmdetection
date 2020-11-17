import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class RoIAttentionPixelToPixelConvFCBBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention_hidden_channels=256,
                 attention_pool_size=2,
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.attention_hidden_channels = attention_hidden_channels
        self.q_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.k_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.v_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.y_conv = nn.Conv2d(attention_hidden_channels, conv_out_channels, kernel_size=1)
        self.attention_pool_size = attention_pool_size

    def init_weights(self):
        super(RoIAttentionPixelToPixelConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule

    def forward(self, x):
        """

        :param x: shape (BS, num_rois, C, H, W)
        :return:
        """
        BS, num_rois, C, H, W = x.shape
        x = x.reshape(BS*num_rois, C, H, W)

        Q = self.q_conv(x)  # (BS*num_rois, attention_hidden_channels, H, W)
        _x = F.max_pool2d(x, self.attention_pool_size, self.attention_pool_size)
        _H, _W = H // self.attention_pool_size, W // self.attention_pool_size
        K = self.k_conv(_x)  # (BS*num_rois, attention_hidden_channels, _H, _W)
        V = self.v_conv(_x)  # (BS*num_rois, attention_hidden_channels, _H, _W)

        Q = Q.permute(0, 2, 3, 1)  # (BS*num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois, H, W, self.attention_hidden_channels)
        Q = Q.reshape(BS, num_rois*H*W, self.attention_hidden_channels)  # (BS, num_rois*H*W, attention_hidden_channels)

        K = K.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        K = K.reshape(BS, num_rois, _H, _W, self.attention_hidden_channels)
        K = K.reshape(BS, num_rois*_H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        V = V.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        V = V.reshape(BS, num_rois, _H, _W, self.attention_hidden_channels)
        V = V.reshape(BS, num_rois*_H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        WEIGHTS = torch.bmm(Q, K.permute(0, 2, 1))  # (BS, num_rois*H*W, num_rois*_H*_W)
        WEIGHTS = torch.softmax(WEIGHTS, dim=2)

        Y = torch.bmm(WEIGHTS, V)  # (BS, num_rois*H*W, attention_hidden_channels)
        Y = Y.reshape(BS, num_rois, H, W, self.attention_hidden_channels)
        Y = Y.reshape(BS*num_rois, H, W, self.attention_hidden_channels)
        Y = Y.permute(0, 3, 1, 2)
        y = self.y_conv(Y)

        x_enhanced = x + y
        return super(RoIAttentionPixelToPixelConvFCBBoxHead, self).forward(x_enhanced)


@HEADS.register_module()
class RoIAttentionPixelToPixelShared2FCBBoxHead(RoIAttentionPixelToPixelConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionPixelToPixelShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionPixelToPixelShared4Conv1FCBBoxHead(RoIAttentionPixelToPixelConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionPixelToPixelShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionPixelToObjectConvFCBBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention_hidden_channels=256,
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.attention_hidden_channels = attention_hidden_channels
        self.score_conv = nn.Conv2d(conv_out_channels, 1, kernel_size=1)
        self.q_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.k_fc = nn.Linear(conv_out_channels, attention_hidden_channels)
        self.v_fc = nn.Linear(conv_out_channels, attention_hidden_channels)
        self.y_conv = nn.Conv2d(attention_hidden_channels, conv_out_channels, kernel_size=1)

    def init_weights(self):
        super(RoIAttentionPixelToObjectConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.k_fc, self.v_fc]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """

        :param x: shape (BS, num_rois, C, H, W)
        :return:
        """
        BS, num_rois, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        Q = self.q_conv(x)  # (BS*num_rois, attention_hidden_channels, H, W)
        scores = self.score_conv(x)  # (BS*num_rois, 1, H, W)
        scores = scores.reshape(-1, 1, H*W)
        scores = torch.softmax(scores, dim=2)
        scores = scores.reshape(-1, 1, H, W)

        # from .get_param import get_count, increment_count, save
        # import os
        # _filename = os.path.join("/tmp/debug/", "weight-{}.pickle".format(get_count()))
        # increment_count()
        # save(_filename, scores.detach().cpu().numpy())

        f_rois = x * scores  # (BS*num_rois, C, H, W)
        f_rois = f_rois.sum(dim=[2, 3])  # (BS * num_rois, C)
        K = self.k_fc(f_rois)  # (BS * num_rois, attention_hidden_channels)
        V = self.v_fc(f_rois)  # (BS * num_rois, attention_hidden_channels)
        K = K.reshape(BS, num_rois, -1)
        V = V.reshape(BS, num_rois, -1)

        Q = Q.permute(0, 2, 3, 1)  # (BS*num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois, H, W, -1)  # (BS, num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois*H*W, -1)  # (BS, num_rois*H*W, attention_hidden_channels) or (BS, num_points, attention_hidden_channels)

        K = K.permute(0, 2, 1)
        weights = torch.bmm(Q, K)  # (BS, num_points, num_rois)
        weights = torch.softmax(weights, dim=2)
        Y = torch.bmm(weights, V)  # (BS, num_points, attention_hidden_channels)
        Y = Y.reshape(BS, num_rois, H, W, -1)
        Y = Y.reshape(-1, H, W, self.attention_hidden_channels)  # (BS*num_rois, H, W, attention_hidden_channels)
        Y = Y.permute(0, 3, 1, 2)  # (BS*num_rois, attention_hidden_channels, H, W)
        y = self.y_conv(Y)

        x_enhanced = x + y
        return super(RoIAttentionPixelToObjectConvFCBBoxHead, self).forward(x_enhanced)


@HEADS.register_module()
class RoIAttentionPixelToObjectShared2FCBBoxHead(RoIAttentionPixelToObjectConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionPixelToObjectShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionPixelToObjectShared4Conv1FCBBoxHead(RoIAttentionPixelToObjectConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionPixelToObjectShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionObjectToObjectConvFCBBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention_hidden_channels=1024,
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.attention_hidden_channels = attention_hidden_channels
        _in_channels = self.in_channels * self.roi_feat_area
        self.q_fc = nn.Linear(_in_channels, attention_hidden_channels)
        self.k_fc = nn.Linear(_in_channels, attention_hidden_channels)
        self.v_fc = nn.Linear(_in_channels, attention_hidden_channels)
        self.y_fc = nn.Linear(attention_hidden_channels, _in_channels)

    def init_weights(self):
        super(RoIAttentionObjectToObjectConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.q_fc, self.k_fc, self.v_fc, self.y_fc]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """

        :param x: shape (BS, num_rois, C, H, W)
        :return:
        """
        BS, num_rois, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)

        _x = x.reshape(-1, C*H*W)  # (BS*num_rois, C*H*W)
        Q = self.q_fc(_x)  # (BS*num_rois, attention_hidden_channels)
        K = self.k_fc(_x)  # (BS*num_rois, attention_hidden_channels)
        V = self.v_fc(_x)  # (BS*num_rois, attention_hidden_channels)

        Q = Q.reshape(BS, num_rois, -1)  # (BS, num_rois, attention_hidden_channels)
        K = K.reshape(BS, num_rois, -1)  # (BS, num_rois, attention_hidden_channels)
        V = V.reshape(BS, num_rois, -1)  # (BS, num_rois, attention_hidden_channels)

        WEIGHTS = torch.bmm(Q, K.permute(0, 2, 1))  # (BS, num_rois, num_rois)
        WEIGHTS = torch.softmax(WEIGHTS, dim=2)

        Y = torch.bmm(WEIGHTS, V)  # (BS, num_rois, attention_hidden_channels)
        Y = Y.reshape(BS*num_rois, -1)  # (BS*num_rois, attention_hidden_channels)
        y = self.y_fc(Y)  # (BS*num_rois, C*H*W)
        y = y.reshape(BS*num_rois, C, H, W)

        x_enhanced = x + y
        return super(RoIAttentionObjectToObjectConvFCBBoxHead, self).forward(x_enhanced)


@HEADS.register_module()
class RoIAttentionObjectToObjectShared2FCBBoxHead(RoIAttentionObjectToObjectConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionObjectToObjectShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionObjectToObjectShared4Conv1FCBBoxHead(RoIAttentionObjectToObjectConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionObjectToObjectShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
