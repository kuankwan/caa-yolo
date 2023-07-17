_base_ = 'yolov8_l_base_city_foggy.py'
data_root = '/home/gsx/.conda/envs/kuan11/DetLAB-master/data/cityscapes/'  # Root path of data
target_root = '/home/gsx/.conda/envs/kuan11/DetLAB-master/data/foggy_cityscapes/'
# Path of val annotation file
val_ann_file = 'annotations/instances_foggy_val.json'
val_data_prefix = 'images/val/'  # Prefix of val image path
class_name = ('person','rider','car','truck','bus',
              'train','motorcycle','bicycle',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

close_mosaic_epochs = 20
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
max_epochs = 100
train_batch_size_per_gpu = 4
train_num_workers = 2
save_epoch_intervals = 1
# validation intervals in stage 1
val_interval_stage2 = 1

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'  # noqa

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

val_evaluator = dict(ann_file=target_root + val_ann_file)
test_evaluator = val_evaluator

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,)

default_hooks = dict(
    checkpoint=dict(interval=save_epoch_intervals, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=1000),
    logger=dict(type='LoggerHook', interval=20))
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
