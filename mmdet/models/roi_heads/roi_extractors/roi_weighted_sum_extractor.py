import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .single_level_roi_extractor import SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class RoIWeightedSumExtractor(SingleRoIExtractor):

    def __init__(self, out_channels, featmap_strides):
        super().__init__({}, out_channels, featmap_strides)
        pass

    def __init2__(self, out_channels, featmap_strides):
        self.roi_layers = nn.ModuleList(
            [RoIWeightedSumLayer(spatial_scale=1 / s) for s in featmap_strides]
        )
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

        self.fp16_enabled = False  # for BaseRoIExtractor
        self.finest_scale = 56  # SingleRoIExtractor

    def build_roi_layers(self, layer_cfg, featmap_strides):
        roi_layers = nn.ModuleList(
            [RoIWeightedSumLayer(spatial_scale=1 / s) for s in featmap_strides])
        return roi_layers

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, score_maps, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois, score_maps[0])

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_, score_maps[i])
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats


class RoIWeightedSumLayer(nn.Module):

    def __init__(self, spatial_scale=1.0):
        super().__init__()
        self.output_size = (1, 1)
        self.spatial_scale = float(spatial_scale)

    def forward(self, input, rois, score_map):
        """
        Args:
            input: NCHW images
            rois: B*5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
            score_map: N1HW mask
        """
        _, C, H, W = input.shape

        rets = []
        for roi in rois:
            mask = input.new_zeros(H, W)
            batch_id, x1, y1, x2, y2 = roi
            batch_id = batch_id.to(torch.int64)
            x1, y1, x2, y2 = torch.round(x1 * self.spatial_scale).to(torch.int64), \
                             torch.round(y1 * self.spatial_scale).to(torch.int64), \
                             torch.round(x2 * self.spatial_scale).to(torch.int64), \
                             torch.round(y2 * self.spatial_scale).to(torch.int64)

            if x1 >= x2 or y1 >= y2:
                ret = input.new_zeros(C, 1, 1)
                rets.append(ret)
                continue

            mask[y1:y2, x1:x2] = 1.0
            mask = mask.detach()
            mask = mask * score_map[batch_id]
            mask_roi = mask[:, y1:y2, x1:x2].flatten(start_dim=1)
            mask_roi = torch.softmax(mask_roi, dim=1).reshape((1, y2-y1, x2-x1))
            mask = input.new_zeros(1, H, W)
            mask[:, y1:y2, x1:x2] = mask_roi

            feat = input[batch_id] * mask
            ret = torch.sum(feat, dim=(1, 2), keepdim=True)
            rets.append(ret)

        rets = torch.stack(rets, dim=0)  # (N, C, 1, 1)
        return rets
