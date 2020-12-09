_base_ = "../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
model = dict(
    roi_head=dict(
        type='BatchRoIHead',
        bbox_head=dict(
            type='RoIAttentionObjectToObjectShared2FCBBoxHead',
            attention_hidden_channels=1024
        )
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = "work_dir/coco/faster_rcnn_r50_fpn_roi_attention_object_to_object_1x_coco/"
