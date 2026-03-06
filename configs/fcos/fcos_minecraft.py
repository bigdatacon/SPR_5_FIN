# _base_ = [
#     './fcos_r50-caffe_fpn_gn-head_1x_coco.py',  # базовая конфигурация FCOS
# ]

# # --- классы Minecraft ---
# classes = (
#     'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
#     'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
# )

# metainfo = dict(
#     classes=classes
# )

# # === Настройка модели ===
# model = dict(
#     type='FCOS',
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=True,
#         style='pytorch'
#     ),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5
#     ),
#     bbox_head=dict(
#         type='FCOSHead',
#         num_classes=len(classes),  # <- 17 классов Minecraft
#         in_channels=256,
#         stacked_convs=4,
#         feat_channels=256,
#         strides=[8, 16, 32, 64, 128],
#         norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
#         loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
#         loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
#         loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
#     )
# )

# # --- размер изображений меньше для CPU ---
# img_scale = (512, 512)

# # --- pipeline для обучения ---
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=img_scale, keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]

# # --- pipeline для тестирования ---
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=img_scale, keep_ratio=True),
#     dict(type='PackDetInputs')
# ]

# # --- dataloader для CPU ---
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
# )

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
# )

# test_dataloader = val_dataloader

# # --- оптимизатор ---
# optim_wrapper = dict(
#     optimizer=dict(
#         type='SGD',
#         lr=0.002,
#         momentum=0.9,
#         weight_decay=0.0001
#     )
# )

# # --- scheduler ---
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1
#     )
# ]

# # --- тренировка ---
# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=1,  # для теста CPU, потом можно увеличить
#     val_interval=1
# )

# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# # --- сохранять checkpoint раз в эпоху ---
# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', interval=1)
# )

# # --- рабочая папка ---
# work_dir = './work_dirs/fcos_minecraft'

# # --- путь к предобученному весу COCO (только backbone будет подгружен) ---
# load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco.pth'


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