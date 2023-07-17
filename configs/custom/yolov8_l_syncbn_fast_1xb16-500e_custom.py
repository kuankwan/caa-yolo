_base_ = 'yolov8_m_syncbn_fast_8xb16-500e_coco.py'

# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512
# data_root = 'data/cityscapes/'  # Root path of data
# target_root = 'data/foggy_cityscapes/'
class_name = ('person','rider','car','truck','bus',
              'train','motorcycle','bicycle',)
num_classes = len(class_name)
mixup_prob = 0.15
close_mosaic_epochs = 100
max_epochs = 100
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
train_batch_size_per_gpu = 4
train_num_workers = 2
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'

# =======================Unmodified in most cases==================
pre_transform = _base_.pre_transform
mosaic_affine_transform = _base_.mosaic_affine_transform
last_transform = _base_.last_transform

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,frozen_stages=1),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels],num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))


train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
