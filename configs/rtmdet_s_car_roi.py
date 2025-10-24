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
data_root = 'Data/coco/'
dataset_type = 'CocoDataset'

# Model - update num_classes to match your dataset
model = dict(
    bbox_head=dict(
        num_classes=6  # Update based on your car damage classes
    )
)

# CarROICrop Transform Configuration
# Using RTMDet-x for vehicle detection, RTMDet-S for damage detection
car_roi_transform = dict(
    type='CarROICrop',
    detector_config='/workspace/mmdetection/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py',
    detector_checkpoint=None,  # Auto-download
    score_threshold=0.3,
    padding_ratio=0.1,
    square_crop=True,
    min_crop_size=100,
    device='cpu',
    fallback_to_original=True,
    vehicle_classes=[2, 3, 4, 6, 8]  # bicycle, car, motorcycle, bus, truck
)

# Training pipeline with CarROICrop
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    car_roi_transform,  # Apply CarROICrop
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False
    ),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        pad_val=(114, 114, 114),
        prob=0.5,
        random_pop=False
    ),
    dict(type='PackDetInputs')
]

# Test pipeline with CarROICrop
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    car_roi_transform,  # Apply CarROICrop
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'crop_bbox')
    )
]

# Dataset configuration
train_dataloader = dict(
    batch_size=32,
    num_workers=10,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=5,
    num_workers=10,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluation configuration
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

# Training configuration
max_epochs = 300
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05)
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
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

# Runtime
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

log_level = 'INFO'
load_from = None
resume = False

