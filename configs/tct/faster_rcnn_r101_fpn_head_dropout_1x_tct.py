_base_ = "../faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py"

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBDropoutBoxHead'
        )
    )
)

dataset_type = 'CocoDataset'

classes = ('normal', 'ascus', 'asch', 'lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em', 'vaginalis', 'monilia',
           'dysbacteriosis_herpes_act', 'ec')

# data_root = '/root/commonfile/TCTAnnotatedData/'
# data_root = '/root/userfolder/datasets/tct/'
data_root = '/home/hezhujun/datasets/tct/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/tct/annotations/train30000-cat10.json',
        img_prefix=data_root),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/tct/annotations/val10000-cat10.json',
        img_prefix=data_root),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/tct/annotations/train30000-cat10.json',
        img_prefix=data_root))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = "work_dir/tct/faster_rcnn_r101_fpn_head_dropout_1x_tct/"
