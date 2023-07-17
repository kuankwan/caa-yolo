_base_ = 'yolov8_s_base_kitti_city.py'
data_root = '/home/gsx/.conda/envs/kuan11/DetLAB-master/data/kitti/'  # Root path of data
target_root = '/home/gsx/.conda/envs/kuan11/DetLAB-master/data/cityscapes/'
class_name = ('car',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

close_mosaic_epochs = 20
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
max_epochs = 300
train_batch_size_per_gpu = 4
train_num_workers = 1
save_epoch_intervals = 5
# validation intervals in stage 2
val_interval_stage2 = 5

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=1),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ))

target_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=target_root,
        metainfo=metainfo,))
#
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=target_root,))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

val_evaluator = dict(ann_file=target_root + 'annotations/val_modified_caronly.json')
test_evaluator = val_evaluator

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,)

default_hooks = dict(
    checkpoint=dict(interval=save_epoch_intervals, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=1000),
    logger=dict(type='LoggerHook', interval=40))
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
