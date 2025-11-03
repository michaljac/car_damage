 # RTMDet-S Config with CarROICrop for Car Damage Detection
# This config uses RTMDet-S as the detection backbone (better accuracy than tiny)
# Vehicle detection still uses RTMDet-x for optimal vehicle cropping

_base_ = '../mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

# Custom imports to load CarROICrop transform
custom_imports = dict(
    imports=['mmdet.datasets.transforms.car_roi_crop'],
    allow_failed_imports=False
)

# Data settings
data_root = '/Data/coco/'
dataset_type = 'CocoDataset'

metainfo = dict(
    classes=('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat'),
    palette=[(220,20,60), (0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]
)

# Model - update num_classes to match your dataset
model = dict(
    bbox_head=dict(
        num_classes=6,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder')
    )
)
# CarROICrop Transform Configuration
# Using RTMDet-x for vehicle detection, RTMDet-S for damage detection

# Training pipeline with CarROICrop
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CarROICrop', vehicle_class_id=7, save_debug=False),
    dict(
        type='CachedMosaic',
        img_scale=(1024, 1024),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False
    ),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    # dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size_divisor=32,
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(
        type='CachedMixUp',
        img_scale=(800, 800),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        pad_val=(114, 114, 114),
        prob=0.3,
        random_pop=False
    ),
    dict(type='PackDetInputs')
]

# Test pipeline with CarROICrop
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CarROICrop', vehicle_class_id=7, save_debug=False),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='Pad',
        size_divisor=32,
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'crop_bbox')
    )
]

# Dataset configuration
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/annotations_train.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline
    
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/annotations_val.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluation configuration
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/annotations_val.json',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

# Training configuration
max_epochs = 180
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',                  
    optimizer=dict(type='AdamW', lr=5e-6, weight_decay=0.05),
    clip_grad=dict(max_norm=10.0),    
    accumulative_counts=4,                   
    loss_scale='dynamic'                     
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=150,
        end=300,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Logging
# --- Logging & Visualization ---
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

# Visualization backend (includes W&B)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend')
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# custom_hooks = [
#     dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
#     dict(type='SyncRandomSizeHook', ratio_range=(14, 26), priority=48)
# ]
# --- W&B Logging Hook ---
# custom_hooks = [
#     dict(
#         type='WandbLoggerHook',
#         init_kwargs=dict(
#             project='car_damage_detection',
#             name='rtmdet_s_car_roi',
#             tags=['car-damage', 'roi-crop', 'rtmdet-s'],
#             notes='RTMDet-S + CarROICrop fine-tuning'
#         ),
#         interval=50,
#         log_checkpoint=True,
#         log_checkpoint_metadata=True,
#         num_eval_images=20
#     )
# ]

# --- Runtime ---
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

log_level = 'INFO'
load_from = None
resume = True


