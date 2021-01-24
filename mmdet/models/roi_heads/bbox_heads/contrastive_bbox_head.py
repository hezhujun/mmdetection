import torch

from .convfc_bbox_head import *
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from .sample_queue import *


@HEADS.register_module()
class ContrastiveConvFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_shared_convs=0, num_shared_fcs=0, num_cls_convs=0, num_cls_fcs=0, num_reg_convs=0,
                 num_reg_fcs=0, conv_out_channels=256, fc_out_channels=1024, conv_cfg=None, norm_cfg=None,
                 queue_pos_max_size=100, queue_neg_max_size=100, temperature_factor=1.0, contrastive_loss_weight=1.0,
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.queue_pos_max_size = queue_pos_max_size
        self.queue_neg_max_size = queue_neg_max_size

        self.tau = temperature_factor
        self.contrastive_loss_weight = contrastive_loss_weight

        set_queue_pos_max_size(queue_pos_max_size)
        set_queue_neg_max_size(queue_neg_max_size)

    def contrastive_loss_for_gt(self, i, catId, similarity_matrix, cat_inds, pos_mask, cat_mask):
        pos_similarities = []
        for other in cat_inds[catId]:
            pos_similarities.append(similarity_matrix[i, other])

        # pos_similarities = sum(pos_similarities)

        neg_similarities = []
        neg_similarities.extend(similarity_matrix[i][~cat_mask])

        neg_similarities = sum(neg_similarities)

        contrastive_losses = []
        for pos_similarity in pos_similarities:
            contrastive_losses.append(-torch.log(pos_similarity / (pos_similarity + neg_similarities)))
        contrastive_losses = sum(contrastive_losses) / len(contrastive_losses)

        return contrastive_losses

        # contrastive_loss = -torch.log(pos_similarities / (pos_similarities + neg_similarities))
        # return contrastive_loss

    def loss(self,
             cls_score,
             bbox_pred,
             obj_feats,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)

                # 对比损失
                bg_class_ind = self.num_classes
                pos_mask = (labels >= 0) & (labels < bg_class_ind)
                gt_mask = torch.zeros_like(pos_mask)
                for idx, flag in enumerate(pos_mask):
                    if torch.any(flag):
                        bbox_target = bbox_targets[idx]
                        if not torch.any(bbox_target.to(torch.bool)):
                            # bbox_target 的4个值全为0，说明是gt
                            gt_mask[idx] = 1

                indices = torch.arange(0, len(pos_mask), dtype=torch.int64, device=pos_mask.device)
                pos_mask = pos_mask.to(torch.bool)
                pos_inds = indices[pos_mask]
                gt_mask = gt_mask.to(torch.bool)
                gt_inds = indices[gt_mask]

                cat_masks = {}
                cat_inds = {}
                for i in range(self.num_classes):
                    cat_masks[i] = labels == i
                    cat_inds[i] = indices[cat_masks[i]]

                # obj_feats已经 L2 normalize
                similarity_matrix = torch.matmul(obj_feats, obj_feats.transpose(0, 1))
                similarity_matrix = torch.exp(similarity_matrix / self.tau)

                contrastive_losses = []
                for gt_idx in gt_inds:
                    catId = int(labels[gt_idx].detach().cpu().numpy())
                    contrastive_losses.append(
                        self.contrastive_loss_for_gt(gt_idx, catId,
                                                     similarity_matrix, cat_inds,
                                                     pos_mask, labels == catId)
                    )
                # print(contrastive_losses)
                contrastive_losses = sum(contrastive_losses) / len(contrastive_losses)
                losses["loss_contrastive"] = contrastive_losses * self.contrastive_loss_weight

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                _x = x
                x = self.relu(fc(x))

            # x_cls = self.shared_fcs[-1](x)
            # x_reg = torch.relu(x_cls)
        # separate branches
        # x = x / torch.norm(x, p=None, dim=1, keepdim=True)
        x_cls = x
        x_reg = x
        # x_cls = x_cls / torch.norm(x_cls, p=None, dim=1, keepdim=True)
        # _x_cls = x_cls

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, _x / torch.norm(_x, p=None, dim=1, keepdim=True)


@HEADS.register_module()
class ContrastiveShared2FCBBoxHead(ContrastiveConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(ContrastiveShared2FCBBoxHead, self).__init__(
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
class ContrastiveShared4Conv1FCBBoxHead(ContrastiveConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(ContrastiveShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
