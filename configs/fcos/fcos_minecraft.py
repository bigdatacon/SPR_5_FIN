

# НА БАЗЕ УЧЕБНОГО ПРИМЕРА FCOS, ДОБАВЛЯЕМ НАСТРОЙКИ ДЛЯ МОДЕЛИ И ДАТАСЕТА MINECRAFT

_base_ = [
    '../_base_/models/fcos_r50-caffe_fpn_gn-head_1x_coco.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# # --- классы Minecraft ---
classes = (
    'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
    'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
)


# --- Модификации ---
model = dict(
    backbone=dict(frozen_stages=2),
    roi_head=dict(
        bbox_head=dict(
            num_classes=17
        )
    )
)

data = dict(
    _delete_=True,   # <<< ВАЖНО
    # samples_per_gpu=4,  - у меня CPU пока оставлю не заполнненым
    train=dict(
        type='CocoDataset',
        ann_file='../../datasets/minecraft/annotations/annotations_train.json',
        img_prefix='../../datasets/minecraft/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
    )
)

optimizer = dict(lr=0.005)
load_from = '../../checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco.pth'
