_base_ = ['../fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py']

# --- Классы Minecraft ---
classes = (
    'bee','chicken','cow','creeper','enderman','fox','frog','ghast',
    'goat','llama','pig','sheep','skeleton','spider','turtle','wolf','zombie'
)

# --- Модель ---
model = dict(
    backbone=dict(frozen_stages=2),
    bbox_head=dict(num_classes=len(classes))
)

# --- Пайплайны (уменьшенные картинки для CPU/GPU) ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs'),
]

# --- Датасеты ---
dataset_type = 'CocoDataset'
data_root = 'datasets/minecraft/'

train_dataloader = dict(
    batch_size=2,  # samples_per_gpu
    num_workers=2,  # workers_per_gpu
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/annotations_train.json',
        data_prefix=dict(img='train/images'),
        metainfo=dict(classes=classes)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/annotations_valid.json',
        data_prefix=dict(img='valid/images'),
        metainfo=dict(classes=classes)
    )
)

test_dataloader = val_dataloader

# --- Метрики ---
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/annotations_valid.json',
    metric='bbox'
)
test_evaluator = val_evaluator

# --- Оптимизатор ---
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001
)

# --- Scheduler ---
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,  # 12 эпох
        by_epoch=True,
        milestones=[6, 9],
        gamma=0.1
    )
]

# --- Тренировка ---
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1,
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1)
)



val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --- Рабочая папка ---
work_dir = './artifacts'

# --- Предобученные веса COCO для backbone (относительный путь) ---
# load_from = '../../checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'

import os
load_from = os.path.join(os.getcwd(), 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth')

# --- FP16 ---
fp16 = dict(loss_scale='dynamic')