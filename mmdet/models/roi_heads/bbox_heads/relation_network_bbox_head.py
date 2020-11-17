import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead


class ObjectRelationModule(nn.Module):

    def __init__(self, df=1024, dk=64, dg=64, Nr=16):
        super().__init__()
        self.df = df
        self.dk = dk
        self.dg = dg
        self.Nr = Nr

        self.q_fc = nn.Linear(df, dk * Nr)
        self.k_fc = nn.Linear(df, dk * Nr)
        self.v_fc = nn.Linear(df, df)
        self.g_fc = nn.Linear(dg, Nr)
        self.y_fc = nn.Linear(df, df)

    def init_weight(self):
        for m in [self.q_fc, self.k_fc, self.v_fc, self.g_fc, self.y_fc]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, position_embedding):
        """

        :param x: (num_rois, feat_dim)
        :param position_embedding: (num_rois, num_rois, dg)
        :return:
        """
        num_rois = x.shape[0]

        position_embedding = position_embedding.reshape(-1, self.dg)
        # (num_rois*num_rois, Nr)
        geometry_w = self.g_fc(position_embedding)
        geometry_w = geometry_w.reshape(num_rois, num_rois, self.Nr)
        geometry_w = torch.relu(geometry_w)
        geometry_w = geometry_w.clamp_min(1e-6)
        # (Nr, num_rois, num_rois)
        geometry_w = geometry_w.permute(2, 0, 1)

        Q = self.q_fc(x)  # (num_rois, dk*Nr)
        K = self.k_fc(x)  # (num_rois, dk*Nr)
        V = self.v_fc(x)  # (num_rois, df)

        Q = Q.reshape(num_rois, self.Nr, self.dk)
        K = K.reshape(num_rois, self.Nr, self.dk)
        V = V.reshape(num_rois, self.Nr, self.df // self.Nr)

        Q = Q.permute(1, 0, 2)  # (Nr, num_rois, dk)
        K = K.permute(1, 2, 0)  # (Nr, dk, num_rois)
        V = V.permute(1, 0, 2)  # (Nr, num_rois, df/Nr)

        weight = torch.bmm(Q, K)  # (Nr, num_rois, num_rois)
        weight_all = torch.log(geometry_w) + weight
        weight_all = torch.softmax(weight_all, dim=2)
        if torch.any(torch.isnan(weight_all)):
            print("weight_all has nan element(s)")
        if torch.any(torch.isinf(weight_all)):
            print("weight_all has inf element(s)")

        # (Nr, num_rois, df/Nr)
        Y = torch.bmm(weight_all, V)
        Y = Y.permute(1, 0, 2)
        Y = Y.reshape(num_rois, self.df)  # (num_rois, df)

        y = self.y_fc(Y)

        return x + y


@HEADS.register_module()
class RelationNetworkBBoxHead(BBoxHead):

    def __init__(self,
                 fc1_dim=1024,
                 fc2_dim=1024,
                 r1=1, r2=1,
                 dk=64, dg=64, Nr=16,
                 *args,
                 **kwargs):
        super(RelationNetworkBBoxHead, self).__init__(*args, **kwargs)

        _in_channels = self.in_channels * self.roi_feat_area
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.fc1 = nn.Linear(_in_channels, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.r1 = r1
        self.r2 = r2
        self.dk = 64
        self.dg = 64
        self.Nr = 16
        self.rm1 = nn.ModuleList([
            ObjectRelationModule(df=fc1_dim, dk=dk, dg=dg, Nr=Nr) for _ in range(r1)
        ])
        self.rm2 = nn.ModuleList([
            ObjectRelationModule(df=fc2_dim, dk=dk, dg=dg, Nr=Nr) for _ in range(r2)
        ])

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.fc2_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = nn.Linear(self.fc2_dim, out_dim_reg)

    def init_weights(self):
        super(RelationNetworkBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.fc1, self.fc2]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        for module in self.rm1:
            if isinstance(module, ObjectRelationModule):
                module.init_weight()

        for module in self.rm2:
            if isinstance(module, ObjectRelationModule):
                module.init_weight()

    @staticmethod
    def extract_position_matrix(bbox):
        """

        :param bbox: (num_bbox, 4)
        :return:
        """
        x1, y1, x2, y2 = bbox[:, 0:1], bbox[:, 1:2], bbox[:, 2:3], bbox[:, 3:4]
        bbox_width = torch.clamp_min(x2 - x1, 1)
        bbox_height = torch.clamp_min(y2 - y1, 1)
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)

        delta_x = center_x - center_x.t()  # (num_bbox, num_bbox)
        delta_x = torch.div(delta_x, bbox_width)
        delta_x = torch.abs(delta_x)
        delta_x = torch.clamp_min(delta_x, 1e-3)
        delta_x = torch.log(delta_x)
        if torch.any(torch.isnan(delta_x)):
            print("delta_x has nan element(s)")
        if torch.any(torch.isinf(delta_x)):
            print("delta_x has inf element(s)")

        delta_y = center_y - center_y.t()  # (num_bbox, num_bbox)
        delta_y = torch.div(delta_y, bbox_width)
        delta_y = torch.abs(delta_y)
        delta_y = torch.clamp_min(delta_y, 1e-3)
        delta_y = torch.log(delta_y)
        if torch.any(torch.isnan(delta_y)):
            print("delta_y has nan element(s)")
        if torch.any(torch.isinf(delta_y)):
            print("delta_y has inf element(s)")

        delta_width = torch.div(bbox_width, bbox_width.t())
        delta_width = torch.log(delta_width)
        if torch.any(torch.isnan(delta_width)):
            print("delta_width has nan element(s)")
        if torch.any(torch.isinf(delta_width)):
            print("delta_width has inf element(s)")

        delta_height = torch.div(bbox_height, bbox_height.t())
        delta_height = torch.log(delta_height)
        if torch.any(torch.isnan(delta_height)):
            print("delta_height has nan element(s)")
        if torch.any(torch.isinf(delta_height)):
            print("delta_height has inf element(s)")

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)
        return position_matrix

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
        """

        :param position_mat: (num_rois, num_rois, 4)
        :param feat_dim:
        :param wave_length:
        :return:
        """
        feat_range = torch.arange(0, feat_dim // 8, \
                                  dtype=position_mat.dtype, device=position_mat.device)
        dim_mat = torch.pow(torch.full((1,), fill_value=wave_length, dtype=position_mat.dtype, \
                                       device=position_mat.device), 8 / feat_dim * feat_range)
        if torch.any(torch.isnan(dim_mat)):
            print("dim_mat has nan element(s)")
        if torch.any(torch.isinf(dim_mat)):
            print("dim_mat has inf element(s)")
        dim_mat = dim_mat.reshape(1, 1, 1, -1)
        position_mat = position_mat * 100
        position_mat = position_mat.unsqueeze(dim=3)
        div_mat = torch.div(position_mat, dim_mat)
        if torch.any(torch.isnan(dim_mat)):
            print("dim_mat has nan element(s)")
        if torch.any(torch.isinf(dim_mat)):
            print("dim_mat has inf element(s)")
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)

        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # (num_rois, num_rois, feat_dim)
        embedding = embedding.flatten(start_dim=2)
        return embedding

    def forward(self, x, rois):
        """

        :param x: (BS, num_rois, C, roi_h, roi_w)
        :param rois: (BS, num_rois, 5)
        :return:
        """
        BS, num_rois, C, H, W = x.shape  # BS, num_rois, C, roi_h, roi_w

        bboxes = rois[:, :, 1:]
        bboxes = bboxes.reshape(-1, 4)
        position_mat = self.extract_position_matrix(bboxes)
        position_embedding = self.extract_position_embedding(position_mat, self.dg)

        x = x.flatten(0, 1)  # BS*num_rois, C, roi_h, roi_w
        x = x.flatten(1)  # BS*num_rois, num_features
        x = F.relu(self.fc1(x))

        for m in self.rm1:
            x = m(x, position_embedding)

        x = F.relu(self.fc2(x))

        for m in self.rm2:
            x = m(x, position_embedding)

        x_cls = x
        x_reg = x

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
