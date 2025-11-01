# Vehicle Damage Detection with Dynamic ROI Cropping

## <img src="" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Quick Start Summary
<div>

**🚀 Get Started in 4 Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/michaljac/car_damage.git
   cd car_damage
   ```

2. **Build the image**
- **Windows:**
```bash
docker build -t car_damage:v1 -f Dockerfile .
```

2. **Build the container**
- **Windows:**

```bash
docker run --rm -v "$(pwd):/workspace" -v "$(pwd)/../Data/car_damage:/Data" car_damage:v1
```

4. 
download dataset zip from google drive.
put it in container so you'll have 
.../car_damage:/workspace
.../Data/damage:/Data

after extract the dataset zip, and rename the folder "Dataset" to "coco", so the structure is:
/Data/coco
where the images are in: /Data/coco/train2017 (val2017, test2017)
and the annotations are in: //Data/coco/annotations

5. preprocess



## <img src="" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Project Structure

```
car_damage/                    # Project root
├── Dockerfile               
├── configs/   
|___|__config.yaml                  # Configs
│   ├──faster_rcnn_car_roi.py                 
│   │__rtmdet_s_car_roi.py   
├── mmdetection/                     
│     |__mmdet/                
│         └── datasets/
|                |__transforms/
|                       |___car_roi_crop.py    # the class "on the fly"
|
├── pipelines/                 # main scripts
│   ├── preprocess.py    
│   └── train.py
|   |__ inference.py
|   |__evaluate.py  
├── tests/                  # Test suite
│   ├── uni_preprocess.py    
│   └── uni_train.py
|   |__ uni_inference.py
|   |__uni_evaluate.py                 
├── utils/                  # helpers
├── workdirs/                 # all the runs (unuploaded to git)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```



## Training

**Option 1: Docker (Recommended - No WSL2 I/O issues)**
```bash
docker run --rm -it \
  -v /workspace:/workspace \
  -v /Data:/Data \
  -w /workspace \
  car_damage:v1 \
  bash -c "PYTHONPATH=/workspace/mmdetection:\$PYTHONPATH python mmdetection/tools/train.py configs/rtmdet_s_car_roi.py --work-dir work_dirs/rtmdet_s_car_roi"
```

**Option 2: Direct (if no I/O issues)**
```bash
cd /workspace && PYTHONPATH=/workspace/mmdetection:$PYTHONPATH python mmdetection/tools/train.py configs/rtmdet_s_car_roi.py --work-dir work_dirs/rtmdet_s_car_roi
```

## What CarROICrop Does

During training, the `CarROICrop` transform:
1. ✅ Finds the vehicle bbox (category_id=7) in each image
2. ✅ Crops the image to the vehicle region  
3. ✅ Adjusts all damage bboxes to the cropped coordinates
4. ✅ Returns original image unchanged if no valid boxes remain (prevents "max fetches" error)

This happens **on-the-fly** during training for each batch!