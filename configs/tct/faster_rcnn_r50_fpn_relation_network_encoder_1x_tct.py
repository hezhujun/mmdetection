_base_ = "../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        _delete_=True,
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        _delete_=True,
        type='RelationNetworkRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RelationNetworkBBoxHeadEncoder',
            in_channels=256,
            roi_feat_size=7,
            num_classes=10,
            fc1_dim=1024,
            fc2_dim=1024,
            r1=1,
            r2=0,
            dk=64,
            dg=64,
            Nr=16,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

dataset_type = 'CocoDataset'

classes = ('normal', 'ascus', 'asch', 'lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em', 'vaginalis', 'monilia',
           'dysbacteriosis_herpes_act', 'ec')

# data_root = '/root/commonfile/TCTAnnotatedData/'
# data_root = '/root/userfolder/datasets/tct/'
data_root = '/home/hezhujun/datasets/tct/'
data = dict(
    samples_per_gpu=1,
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
        ann_file='data/tct/annotations/test10000-cat10.json',
        img_prefix=data_root))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
work_dir = "work_dir/tct/faster_rcnn_r50_fpn_relation_network_encoder_1x_tct/"
