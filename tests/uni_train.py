"""
Unified Training Test for CarROICrop + RTMDet
Tests the complete training pipeline with on-the-fly vehicle cropping.

This script:
1. Validates dataset paths (train + val)
2. Validates config parameters
3. Tests CarROICrop transform loading
4. Runs training for a few epochs (cuda mode for testing)
5. Monitors loss and mAP via WandB

Usage:
    python tests/uni_train.py
"""

import os
import sys

# Add paths - LOCAL mmdetection FIRST to use our custom CarROICrop
sys.path.insert(0, '/workspace/mmdetection')
sys.path.insert(0, '/workspace')

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

# Import datasets.transforms to ensure CarROICrop is registered
from mmdet.datasets import transforms
# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Config file
CONFIG_FILE = 'configs/rtmdet_s_car_roi.py'  # RTMDet-S with CarROICrop

# Working directory for outputs
WORK_DIR = 'work_dirs/test_train_car_roi'

# Training settings
NUM_EPOCHS = 5  # Just a few epochs for testing
DEVICE = 'cpu'  # CPU training
BATCH_SIZE = 2  # Smaller batch size for CPU
NUM_WORKERS = 2  # Number of workers for CPU

# Resume from checkpoint (if exists)
RESUME = False

# WandB settings
USE_WANDB = True
WANDB_PROJECT = 'car-damage-detection'
WANDB_RUN_NAME = 'rtmdet-tiny-carroicrop-test'

# Expected dataset structure
EXPECTED_STRUCTURE = """
Expected dataset structure:
/Data/coco/
├── train2017/          # Training images
├── val2017/            # Validation images
├── annotations/
│   ├── annotations_train.json
│   └── annotations_val.json
"""


# ============================================================================
# Validation Functions
# ============================================================================

def validate_dataset_paths(cfg):
    """Validate that dataset paths exist."""
    print("\n" + "="*70)
    print("VALIDATING DATASET PATHS")
    print("="*70)
    
    data_root = cfg.data_root
    print(f"Data root: {data_root}")
    
    # Check data root
    if not os.path.exists(data_root):
        print(f" ERROR: Data root not found: {data_root}")
        print(EXPECTED_STRUCTURE)
        return False
    print(f"✓ Data root exists")
    
    # Check train images
    train_img_dir = os.path.join(data_root, 'train2017')
    if not os.path.exists(train_img_dir):
        print(f" ERROR: Train images not found: {train_img_dir}")
        return False
    train_img_count = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"✓ Train images: {train_img_dir} ({train_img_count} images)")
    
    # Check val images
    val_img_dir = os.path.join(data_root, 'val2017')
    if not os.path.exists(val_img_dir):
        print(f" ERROR: Val images not found: {val_img_dir}")
        return False
    val_img_count = len([f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"✓ Val images: {val_img_dir} ({val_img_count} images)")
    
    # Check annotations
    ann_dir = os.path.join(data_root, 'annotations')
    if not os.path.exists(ann_dir):
        print(f" ERROR: Annotations dir not found: {ann_dir}")
        return False
    
    train_ann = os.path.join(ann_dir, 'annotations_train.json')
    val_ann = os.path.join(ann_dir, 'annotations_val.json')
    
    if not os.path.exists(train_ann):
        print(f" ERROR: Train annotations not found: {train_ann}")
        return False
    print(f"✓ Train annotations: {train_ann}")
    
    if not os.path.exists(val_ann):
        print(f"ERROR: Val annotations not found: {val_ann}")
        return False
    print(f"✓ Val annotations: {val_ann}")
    
    print("✅ All dataset paths validated!")
    return True


def validate_carroicrop_config(cfg):
    """Validate that CarROICrop is in the pipeline."""
    print("\n" + "="*70)
    print("VALIDATING CARROICROP CONFIGURATION")
    print("="*70)
    
    # Check train pipeline
    train_pipeline = cfg.train_dataloader.dataset.pipeline
    has_carroicrop = any(t.get('type') == 'CarROICrop' for t in train_pipeline)
    
    if not has_carroicrop:
        print(" ERROR: CarROICrop not found in train_pipeline!")
        return False
    
    print("✓ CarROICrop found in train_pipeline")
    
    # Find and display CarROICrop config
    for transform in train_pipeline:
        if transform.get('type') == 'CarROICrop':
            print("\nCarROICrop Configuration:")
            print(f"  - vehicle_class_id: {transform.get('vehicle_class_id', 7)}")
            print(f"  - save_debug: {transform.get('save_debug', False)}")
            break
    
    # Check val pipeline
    val_pipeline = cfg.val_dataloader.dataset.pipeline
    has_carroicrop = any(t.get('type') == 'CarROICrop' for t in val_pipeline)
    
    if not has_carroicrop:
        print("⚠️  WARNING: CarROICrop not found in val_pipeline (evaluation won't use cropping)")
    else:
        print("✓ CarROICrop found in val_pipeline")
    
    print("✅ CarROICrop configuration validated!")
    return True


def test_carroicrop_import():
    """Test that CarROICrop can be imported."""
    print("\n" + "="*70)
    print("TESTING CARROICROP IMPORT")
    print("="*70)
    
    try:
        from mmdet.datasets.transforms import CarROICrop
        print("✓ CarROICrop imported successfully from mmdet.datasets.transforms")
        
        # Try to instantiate with simple parameters
        transform = CarROICrop(
            vehicle_class_id=7,
            save_debug=False
        )
        print("✓ CarROICrop instantiated successfully")
        print(f"  {transform}")
        
        print("✅ CarROICrop import test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to import CarROICrop: {e}")
        return False


def display_training_parameters(cfg):
    """Display key training parameters."""
    print("\n" + "="*70)
    print("TRAINING PARAMETERS")
    print("="*70)
    
    print(f"Model: RTMDet-tiny")
    print(f"Number of classes: {cfg.model.bbox_head.num_classes}")
    print(f"Max epochs: {cfg.max_epochs}")
    print(f"Base learning rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Batch size (train): {cfg.train_dataloader.batch_size}")
    print(f"Batch size (val): {cfg.val_dataloader.batch_size}")
    print(f"Device: {cfg.device if hasattr(cfg, 'device') else 'cuda (default)'}")
    print(f"Work directory: {cfg.work_dir}")
    
    print("\nDataset:")
    print(f"  Train: {cfg.train_dataloader.dataset.data_root}")
    print(f"  Val: {cfg.val_dataloader.dataset.data_root}")
    
    print("\nVisualization:")
    if hasattr(cfg, 'visualizer') and hasattr(cfg.visualizer, 'vis_backends'):
        for backend in cfg.visualizer.vis_backends:
            backend_type = backend.get('type', 'Unknown')
            print(f"  - {backend_type}")
            if backend_type == 'WandbVisBackend':
                print(f"    Project: {backend.get('init_kwargs', {}).get('project', 'N/A')}")
                print(f"    Name: {backend.get('init_kwargs', {}).get('name', 'N/A')}")
    else:
        print("  - TensorBoard (default)")


# ============================================================================
# Main Training Function
# ============================================================================

def run_training_test():
    """Run training test with validation."""
    
    print("="*70)
    print("UNIFIED TRAINING TEST - CARROICROP + RTMDET-TINY")
    print("="*70)
    
    # Load config
    print(f"\n📄 Loading config: {CONFIG_FILE}")
    cfg = Config.fromfile(CONFIG_FILE)
    
    # Override settings for testing
    cfg.max_epochs = NUM_EPOCHS
    cfg.device = DEVICE
    cfg.work_dir = WORK_DIR
    cfg.train_dataloader.batch_size = BATCH_SIZE
    cfg.val_dataloader.batch_size = BATCH_SIZE
    cfg.train_dataloader.num_workers = NUM_WORKERS  # CPU workers
    cfg.val_dataloader.num_workers = NUM_WORKERS
    
    # CarROICrop doesn't need device parameter - it works with annotations only
    # (no internal detector in the simple version)
    
    # Configure WandB if enabled
    if USE_WANDB:
        cfg.visualizer = dict(
            type='DetLocalVisualizer',
            vis_backends=[
                dict(type='LocalVisBackend'),
                dict(
                    type='WandbVisBackend',
                    init_kwargs=dict(
                        project=WANDB_PROJECT,
                        name=WANDB_RUN_NAME,
                        # Don't pass config dict - MMDetection will log config automatically
                    )
                )
            ]
        )
        print(f"✓ WandB configured: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
    
    # Resume if requested
    if RESUME:
        cfg.resume = True
        print("✓ Resume enabled")
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Run validations
    if not validate_dataset_paths(cfg):
        print("\n❌ Dataset validation failed! Please fix the issues above.")
        return False
    
    if not test_carroicrop_import():
        print("\n❌ CarROICrop import test failed!")
        return False
    
    if not validate_carroicrop_config(cfg):
        print("\n❌ CarROICrop configuration validation failed!")
        return False
    
    # Display parameters
    display_training_parameters(cfg)
    
    # Confirm to start training
    print("\n" + "="*70)
    print("READY TO START TRAINING")
    print("="*70)
    print(f"This will train for {NUM_EPOCHS} epochs on {DEVICE}")
    print(f"Results will be saved to: {WORK_DIR}")
    if USE_WANDB:
        print(f"WandB logging: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
    print("="*70)
    
    # Build runner
    print("\n⏳ Building runner...")
    runner = Runner.from_cfg(cfg)
    
    print("\n🚀 Starting training...")
    print("="*70)
    
    # Start training
    try:
        runner.train()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results saved to: {WORK_DIR}")
        print(f"Best checkpoint: {os.path.join(WORK_DIR, 'best_coco_bbox_mAP_50_epoch_*.pth')}")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TRAINING FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    success = run_training_test()
    sys.exit(0 if success else 1)


