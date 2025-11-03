_base_ = '/car_damage/configs/rtmdet_s_car_roi.py'
load_from = 'work_dirs/rtmdet_s_car_roi/epoch_150.pth'

dataset_type = 'CocoDataset'
data_root = '/Data/coco/'

metainfo = dict(
    classes=('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat'),
    palette=[(220,20,60), (0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]
)

# Test pipeline for EVALUATION (with GT)
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

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
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

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/annotations_val.json',
    metric='bbox'
)

test_cfg = dict(type='TestLoop')
