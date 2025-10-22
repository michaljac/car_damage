_base_ = 'mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

# ===== Model (set your number of damage classes) =====
model = dict(
    bbox_head=dict(num_classes=6)
)

# ===== Dataset & classes =====
dataset_type = 'CocoDataset'
data_root = 'data/car_damage/'
metainfo = dict(classes=('dent','scratch','crack','glass shatter','lamp broken','tire flat'))

# ===== Pipelines =====
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # On-the-fly crop around GT car box
    dict(
        type='CropCarROI',
        mode='train',
        # car_class_id=<CAR_CLASS_ID_IN_YOUR_LABELS>,  # e.g., 2
        expand_ratio=1.15,
        keep_if_no_car=True
    ),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

# For val/test, we *also* crop, but by prepass detector
# Provide a small, fast car-detector (RTMW/RTMDet tiny) trained on cars on full frames.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CropCarROI',
        mode='test',
        expand_ratio=1.15,
        keep_if_no_car=True,
        prepass=dict(
            det_config='configs/car_damage/rtmw_tiny_cars_fullimg.py',
            det_ckpt='work_dirs/rtmw_tiny_cars_fullimg/best_coco_bbox_mAP_epoch_XX.pth',
            device='cuda:0',
            score_thr=0.30,
            # optional: restrict labels that count as "car" in the prepass
            # car_label_ids=[<id(s) in the prepass detector>]
        )
    ),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # keep annotations step only for val evaluation (COCO AP)
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root+'annotations/instances_val.json', metric='bbox')
test_evaluator = val_evaluator

# ===== Schedule / runtime tweaks =====
train_cfg = dict(max_epochs=100)
optim_wrapper = dict(optimizer=dict(lr=0.004))
default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=3))
