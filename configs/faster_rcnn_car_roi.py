# Faster R-CNN Config with CarROICrop for Car Damage Detection
# This config uses Faster R-CNN with ResNet-50 backbone
# Vehicle detection uses RTMDet-x for optimal vehicle cropping

_base_ = [
    '../mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../mmdetection/configs/_base_/datasets/coco_detection.py',
    '../mmdetection/configs/_base_/schedules/schedule_1x.py',
    '../mmdetection/configs/_base_/default_runtime.py'
]

# Custom imports to load CarROICrop transform
custom_imports = dict(
    imports=['mmdetection.mmdet.datasets.transforms.car_roi_crop'],
    allow_failed_imports=False
)

# Data settings
data_root = '/Data/coco/'

# Model - update num_classes to match your dataset
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=6  # Update based on your car damage classes
        )
    )
)

# CarROICrop Transform Configuration
# Using RTMDet-x for vehicle detection, Faster R-CNN for damage detection
car_roi_transform = dict(
    type='CarROICrop',
    detector_config='mmdetection/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py',
    detector_checkpoint=None,  # Auto-download
    score_threshold=0.3,
    padding_ratio=0.1,
    square_crop=True,
    min_crop_size=100,
    device='cuda',
    fallback_to_original=True,
    vehicle_classes=[2, 3, 4, 6, 8]  # bicycle, car, motorcycle, bus, truck
)

# Training pipeline with CarROICrop
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    car_roi_transform,  # Apply CarROICrop
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# Test pipeline with CarROICrop
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    car_roi_transform,  # Apply CarROICrop
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'crop_bbox')
    )
]

# Dataset configuration
train_dataloader = dict(
    batch_size=16,  # Faster R-CNN uses smaller batch size
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/annotations_train.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
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

# Training configuration - Faster R-CNN typically trains for 12 epochs (1x schedule)
max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1
    )
]

# Logging
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
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

# NOTE: Faster R-CNN is slower than RTMDet but may provide better accuracy
# Consider using 3x schedule (36 epochs) for even better results

