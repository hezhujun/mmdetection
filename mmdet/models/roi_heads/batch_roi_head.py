from .standard_roi_head import StandardRoIHead
from ..builder import HEADS


@HEADS.register_module()
class BatchRoIHead(StandardRoIHead):

    def _bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        BS = x[0].size(0)
        num_rois, channels, roi_h, roi_w = bbox_feats.shape
        bbox_feats = bbox_feats.reshape((BS, -1, channels, roi_h, roi_w))
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
