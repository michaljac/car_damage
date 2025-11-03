_base_ = '/car_damage/configs/rtmdet_s_car_roi.py'

# === Inference settings ===
# Path to trained checkpoint
load_from = 'work_dirs/rtmdet_s_car_roi/epoch_150.pth'

# Use the same metainfo and transforms as training
dataset_type = 'CocoDataset'
data_root = '/Data/coco/'  # default; can be overridden via CLI arg
metainfo = dict(
    classes=('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat'),
    palette=[(220,20,60), (0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]
)

# Model - update num_classes to match your dataset
model = dict(
    bbox_head=dict(
        num_classes=6  # Update based on your car damage classes
    )
)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
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

# Dummy dataset that loads arbitrary images for inference
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/annotations_test.json',  # not actually needed if only images are given
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/annotations_test.json',
    metric='bbox'
)

# Disable training
train_dataloader = None
val_dataloader = None
train_cfg = None
val_cfg = None
test_cfg = dict(type='TestLoop')

# Inference thresholds
conf_thr = 0.3  # Confidence threshold
iou_thr = 0.5   # NMS IoU threshold
classes_names = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']