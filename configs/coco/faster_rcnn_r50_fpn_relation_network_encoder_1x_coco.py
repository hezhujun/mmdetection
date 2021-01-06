_base_ = "../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
model = dict(
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
            num_classes=80,
            fc1_dim=1024,
            fc2_dim=1024,
            r1=0,
            r2=0,
            dk=1024,
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
work_dir = "work_dir/coco/faster_rcnn_r50_fpn_relation_network_encoder_1x_coco/"
