# Car Damage Detection with CarROICrop

A complete object detection pipeline for car damage detection, featuring **CarROICrop** - an on-the-fly vehicle detection and cropping system built on MMDetection.

## 🎯 Features

- ✅ **On-the-fly Vehicle ROI Cropping**: Automatically detects and crops to vehicles during training/inference
- ✅ **Multiple Backbone Support**: RTMDet-tiny, RTMDet-S, Faster R-CNN (easy to add more)
- ✅ **Clean Project Structure**: Organized configs, pipelines, and plugins
- ✅ **Dataset Flexibility**: Works with COCO format, supports train/val/test splits
- ✅ **Production Ready**: Complete training, evaluation, and inference scripts

## 📁 Project Structure

```
/workspace/
├── mmdetection/                    # MMDetection library
│   └── mmdet/
│       └── datasets/
│           └── transforms/
│               └── car_roi_crop.py # CarROICrop plugin
│
├── configs/                        # Model configurations
│   ├── rtmdet_tiny_car_roi.py     # RTMDet-tiny (fast, baseline)
│   ├── rtmdet_s_car_roi.py        # RTMDet-S (balanced)
│   ├── faster_rcnn_car_roi.py     # Faster R-CNN (accurate)
│   └── README.md                  # Config documentation
│
├── pipelines/                      # Application scripts
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   ├── inference.py               # Inference script (with CarROICrop)
│   └── preprocess.py              # Data preprocessing
│
├── examples/                       # Usage examples
│   └── simple_inference_example.py
│
├── work_dirs/                      # Training outputs (checkpoints, logs)
├── Data/                           # Dataset directory
│   └── coco/
│       ├── train2017/
│       ├── val2017/
│       └── annotations/
│
├── requirements.txt
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo>
cd workspace

# Install dependencies
pip install -r requirements.txt

# Install MMDetection
cd mmdetection
pip install -v -e .
cd ..
```

### 2. Prepare Dataset

Organize your COCO dataset:

```bash
Data/coco/
├── train2017/          # Training images
├── val2017/            # Validation images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Optional: Run preprocessing to clean dataset:

```bash
python pipelines/preprocess.py \
    --data-root Data/coco/ \
    --splits train val
```

### 3. Train

Choose a config and start training:

```bash
# Fast baseline (RTMDet-tiny)
python pipelines/train.py configs/rtmdet_tiny_car_roi.py

# Balanced (RTMDet-S)
python pipelines/train.py configs/rtmdet_s_car_roi.py

# Best accuracy (Faster R-CNN)
python pipelines/train.py configs/faster_rcnn_car_roi.py
```

### 4. Evaluate

```bash
python pipelines/evaluate.py \
    configs/rtmdet_tiny_car_roi.py \
    work_dirs/rtmdet_tiny_car_roi/epoch_300.pth
```

### 5. Inference

```bash
# On validation set
python pipelines/inference.py \
    configs/rtmdet_tiny_car_roi.py \
    work_dirs/rtmdet_tiny_car_roi/epoch_300.pth \
    --dataset val

# On custom images
python pipelines/inference.py \
    configs/rtmdet_tiny_car_roi.py \
    work_dirs/rtmdet_tiny_car_roi/epoch_300.pth \
    --dataset custom \
    --custom-input /path/to/images/
```

## 🔧 CarROICrop System

### What is CarROICrop?

CarROICrop is a custom MMDetection transform that:
1. **Detects vehicles** (car, bicycle, motorcycle, bus, truck) using RTMDet-x
2. **Selects the largest vehicle** by bounding box area
3. **Crops the image** with padding to focus on the vehicle
4. **Adjusts all annotations** to the new cropped coordinates
5. Works **on-the-fly** during training and inference (no preprocessing!)

### Why Use CarROICrop?

- ✅ **Focus on relevant regions**: Removes background clutter
- ✅ **Better accuracy**: Model learns from vehicle-focused crops
- ✅ **Automatic**: No manual cropping or preprocessing
- ✅ **Flexible**: Falls back to original image if no vehicle found

### Architecture

```
Image → LoadImage → LoadAnnotations 
  ↓
CarROICrop (RTMDet-x detector)
  ├─→ Detect vehicles
  ├─→ Select largest
  ├─→ Crop with padding
  └─→ Adjust all bboxes
  ↓
Augmentations (Resize, Flip, etc.)
  ↓
Your Model (RTMDet-tiny/S or Faster R-CNN)
  ↓
Predictions
```

### Vehicle Detection vs Damage Detection

| Component | Purpose | Model |
|-----------|---------|-------|
| **Vehicle Detection** | Find and crop to vehicles | RTMDet-x (best accuracy) |
| **Damage Detection** | Detect damage on vehicles | Your choice (tiny/S/Faster R-CNN) |

**Key Insight**: We use RTMDet-x for vehicle detection (accurate) while you can train with RTMDet-tiny (fast) for damage detection!

## 📊 Model Comparison

### Available Backbones

| Model | Speed | Accuracy | Training Time | Best For |
|-------|-------|----------|---------------|----------|
| **RTMDet-tiny** | ⚡⚡⚡ | ⭐⭐⭐ | ~8 hours | Quick experiments |
| **RTMDet-S** | ⚡⚡ | ⭐⭐⭐⭐ | ~12 hours | Production (balanced) |
| **Faster R-CNN** | ⚡ | ⭐⭐⭐⭐⭐ | ~6 hours* | Maximum accuracy |

\* Faster R-CNN trains for only 12 epochs vs 300 for RTMDet

### Comparing Backbones

Train and evaluate multiple models:

```bash
# Train all
for config in configs/*_car_roi.py; do
    python pipelines/train.py $config
done

# Evaluate all
python pipelines/evaluate.py configs/rtmdet_tiny_car_roi.py \
    work_dirs/rtmdet_tiny_car_roi/epoch_300.pth

python pipelines/evaluate.py configs/rtmdet_s_car_roi.py \
    work_dirs/rtmdet_s_car_roi/epoch_300.pth

python pipelines/evaluate.py configs/faster_rcnn_car_roi.py \
    work_dirs/faster_rcnn_car_roi/epoch_12.pth
```

Results will show:
- mAP (mean Average Precision)
- Per-class AP
- Inference speed
- Model size

## 🎓 Usage Examples

### Training with Custom Data

1. **Update config**:
   ```python
   # In configs/rtmdet_tiny_car_roi.py
   data_root = '/path/to/your/data/'
   
   model = dict(
       bbox_head=dict(
           num_classes=10  # Your number of classes
       )
   )
   ```

2. **Train**:
   ```bash
   python pipelines/train.py configs/rtmdet_tiny_car_roi.py
   ```

### Inference on Different Datasets

```bash
# On training set (for debugging)
python pipelines/inference.py CONFIG CHECKPOINT \
    --dataset train --max-images 100

# On validation set
python pipelines/inference.py CONFIG CHECKPOINT \
    --dataset val

# On test set
python pipelines/inference.py CONFIG CHECKPOINT \
    --dataset test

# On custom images
python pipelines/inference.py CONFIG CHECKPOINT \
    --dataset custom --custom-input /path/to/images/
```

### Disabling CarROICrop

If you want to run without vehicle cropping:

```bash
python pipelines/inference.py CONFIG CHECKPOINT \
    --dataset val --no-car-roi
```

### Simple 3-Line API

```python
from mmdet.apis import init_detector, inference_detector
import mmcv

# Initialize (CarROICrop applies automatically if in config!)
model = init_detector('configs/rtmdet_tiny_car_roi.py', 'checkpoint.pth')

# Inference
img = mmcv.imread('image.jpg')
result = inference_detector(model, img)
```

See `examples/simple_inference_example.py` for more details.

## ⚙️ Configuration

### Key Parameters

#### CarROICrop Settings

```python
car_roi_transform = dict(
    type='CarROICrop',
    detector_config='path/to/rtmdet_x_config.py',  # Vehicle detector
    detector_checkpoint=None,                       # Auto-download
    score_threshold=0.3,                           # Vehicle confidence
    padding_ratio=0.1,                             # 10% padding around bbox
    square_crop=True,                              # Make crop square
    min_crop_size=100,                             # Min crop dimensions
    device='cpu',                               # GPU device
    fallback_to_original=True,                     # Use original if no vehicle
    vehicle_classes=[2, 3, 4, 6, 8]               # COCO vehicle IDs
)
```

#### Training Settings

```python
# Batch size (adjust based on GPU memory)
train_dataloader = dict(
    batch_size=32,  # RTMDet: 32-64, Faster R-CNN: 2-4
    num_workers=10
)

# Training epochs
max_epochs = 300  # RTMDet: 300, Faster R-CNN: 12-36

# Learning rate
optim_wrapper = dict(
    optimizer=dict(lr=0.004)  # Adjust based on batch size
)
```

### Data Paths

Update these in your config:

```python
data_root = 'Data/coco/'  # Your dataset root

train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')
    )
)
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size in config
train_dataloader = dict(batch_size=16)  # or 8
```

### CarROICrop Not Applying

Check that it's in the pipeline:

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CarROICrop', ...),  # Must be here!
    # ... rest of pipeline
]
```

### No Vehicles Detected

Lower the threshold:

```python
car_roi_transform = dict(
    score_threshold=0.2,  # Lower from 0.3
    # ...
)
```

### Training is Slow

- Use RTMDet-tiny instead of S/Faster R-CNN
- Reduce `num_workers`
- Check if GPU is being used: `nvidia-smi`

## 📚 Documentation

- **Configs**: See `configs/README.md` for detailed config documentation
- **CarROICrop**: See `mmdetection/mmdet/datasets/transforms/car_roi_crop.py` for implementation
- **Examples**: See `examples/` for usage examples
- **MMDetection**: https://mmdetection.readthedocs.io/

## 🤝 Contributing

To add a new backbone:

1. Create config: `configs/mymodel_car_roi.py`
2. Add CarROICrop to pipeline (copy from existing configs)
3. Update `configs/README.md` with model info
4. Test training: `python pipelines/train.py configs/mymodel_car_roi.py`

## 📝 License

This project uses MMDetection (Apache 2.0 License).

## 🙏 Acknowledgments

- [MMDetection](https://github.com/open-mmlab/mmdetection) for the excellent framework
- [RTMDet](https://arxiv.org/abs/2212.07784) for the efficient detector
- COCO dataset for training data

---

**Ready to train! 🚗💥**

For questions or issues, check the documentation in `configs/README.md` or review the example scripts in `examples/`.

