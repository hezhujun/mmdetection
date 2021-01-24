import json
from pycocotools.coco import COCO

if __name__ == '__main__':
    path = "data/tct/annotations/train30000-cat10.json"
    with open(path, "r") as f:
        data = json.load(f)

    coco = COCO(path)
    imgIds = coco.getImgIds()
    imgIds = imgIds[:10]
    imgs = coco.loadImgs(imgIds)
    data["images"] = imgs

    annIds = coco.getAnnIds(imgIds)
    anns = coco.loadAnns(annIds)
    data["annotations"] = anns

    print(json.dumps(data, indent=2))

    with open("data/tct/annotations/train10-cat10.json", "w") as f:
        json.dump(data, f, indent=2)