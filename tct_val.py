import os
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annotationFile = "data/tct/annotations/val10000-cat10.json"
predictionFile = "/home/hezhujun/tools/tide/data/tct/annotations/faster_rcnn_r50_fpn_1x_tct_20201231_1_score_thr=0.05.bbox.json"

if __name__ == '__main__':
    gt = COCO(annotationFile)
    dt = gt.loadRes(predictionFile)

    imgIds = sorted(gt.getImgIds())
    catIds = sorted(gt.getCatIds())
    catInfos = gt.loadCats(catIds)

    eval = COCOeval(gt, dt, "bbox")
    eval.params.imgIds = imgIds
    print("evaluation in all categories")
    eval.params.catIds = catIds
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    for stat in eval.stats:
        print("{:.3f}\t".format(stat), end="")
    print()

    print("evaluation in all categories without normal")
    eval.params.catIds = catIds[1:]
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    for stat in eval.stats:
        print("{:.3f}\t".format(stat), end="")
    print()

    for catId in catIds[1:]:
        print("evaluation in {}".format(catInfos[catId]["name"]))
        eval.params.catIds = catId
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        for stat in eval.stats:
            print("{:.3f}\t".format(stat), end="")
        print()
