import torch
from .bbox_head import *


@HEADS.register_module()
class TCTBBoxHead(BBoxHead):

    def __init__(self, with_avg_pool=False, with_cls=True, with_reg=True, roi_feat_size=7, in_channels=256,
                 num_classes=10, bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                clip_border=True,
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]), reg_class_agnostic=False, reg_decoded_bbox=False, loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0), loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super().__init__(with_avg_pool, with_cls, with_reg, roi_feat_size, in_channels, num_classes, bbox_coder,
                         reg_class_agnostic, reg_decoded_bbox, loss_cls, loss_bbox)

        # if self.with_cls:
        #     # tct的normal类当成背景类
        #     self.fc_cls = nn.Linear(in_channels, num_classes)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        # tct的normal类当成背景类
        labels[labels == 0] = self.num_classes

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
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 1~self.num_classes-1 are FG, 0 is BG
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

    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    # def get_bboxes(self,
    #                rois,
    #                cls_score,
    #                bbox_pred,
    #                img_shape,
    #                scale_factor,
    #                rescale=False,
    #                cfg=None):
    #     if isinstance(cls_score, list):
    #         cls_score = sum(cls_score) / float(len(cls_score))
    #     print(cls_score)
    #     scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
    #
    #     # scores (N, num_class) -> (N, num_class+1)
    #     # multiclass_nms()要求scores的格式是(N, num_class+1)
    #     scores = torch.cat([scores, scores[:, 0:1]], dim=1)
    #
    #     if bbox_pred is not None:
    #         bboxes = self.bbox_coder.decode(
    #             rois[:, 1:], bbox_pred, max_shape=img_shape)
    #     else:
    #         bboxes = rois[:, 1:].clone()
    #         if img_shape is not None:
    #             bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
    #             bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
    #
    #     if rescale and bboxes.size(0) > 0:
    #         if isinstance(scale_factor, float):
    #             bboxes /= scale_factor
    #         else:
    #             scale_factor = bboxes.new_tensor(scale_factor)
    #             bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
    #                       scale_factor).view(bboxes.size()[0], -1)
    #
    #     if cfg is None:
    #         return bboxes, scores
    #     else:
    #         det_bboxes, det_labels = multiclass_nms(bboxes, scores,
    #                                                 cfg.score_thr, cfg.nms,
    #                                                 cfg.max_per_img)
    #
    #         # print(det_bboxes)
    #         print(det_labels)
    #         return det_bboxes, det_labels