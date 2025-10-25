"""
Simple Training Script for Car Damage Detection with CarROICrop
Uses MMDetection framework with WandB logging - No argparse!
"""

import os
import sys

# Add paths to load custom transforms and mmdetection
sys.path.insert(0, '/workspace')
sys.path.insert(0, 'mmdetection')

from mmengine.config import Config
from mmengine.runner import Runner


# ============================================================================
# CONFIGURATION - Edit these values directly
# ============================================================================

# Config file path
CONFIG_FILE = 'configs/rtmdet_tiny_car_roi.py'

# Working directory for checkpoints and logs
WORK_DIR = 'work_dirs/rtmdet_car_damage'

# Training settings
NUM_EPOCHS = 50  # Number of training epochs
DEVICE = 'cuda'  # Use GPU
BATCH_SIZE = 16  # Training batch size (adjust based on GPU memory)
NUM_WORKERS = 4  # Data loading workers

# Resume from checkpoint
RESUME = False  # Set to True to resume from last checkpoint

# WandB settings
USE_WANDB = True
WANDB_PROJECT = 'car-damage-detection'
WANDB_RUN_NAME = 'rtmdet-tiny-carroicrop'


# ============================================================================
# Main Training Function
# ============================================================================

def train():
    """Train RTMDet detector for car damage detection."""
    
    print("=" * 80)
    print("CAR DAMAGE DETECTION TRAINING")
    print("=" * 80)
    print(f"Config:       {CONFIG_FILE}")
    print(f"Work dir:     {WORK_DIR}")
    print(f"Device:       {DEVICE}")
    print(f"Epochs:       {NUM_EPOCHS}")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Resume:       {RESUME}")
    print(f"WandB:        {USE_WANDB}")
    if USE_WANDB:
        print(f"  Project:    {WANDB_PROJECT}")
        print(f"  Run name:   {WANDB_RUN_NAME}")
    print("=" * 80)
    
    # Load configuration
    print("\n Loading config...")
    cfg = Config.fromfile(CONFIG_FILE)
    
    # Override settings
    cfg.work_dir = WORK_DIR
    cfg.device = DEVICE
    
    # Override epochs (defined in config at max_epochs and train_cfg.max_epochs)
    cfg.max_epochs = NUM_EPOCHS
    cfg.train_cfg.max_epochs = NUM_EPOCHS
    
    # Override batch size and workers
    cfg.train_dataloader.batch_size = BATCH_SIZE
    cfg.val_dataloader.batch_size = BATCH_SIZE
    cfg.train_dataloader.num_workers = NUM_WORKERS
    cfg.val_dataloader.num_workers = NUM_WORKERS
    cfg.train_dataloader.persistent_workers = (NUM_WORKERS > 0)
    cfg.val_dataloader.persistent_workers = (NUM_WORKERS > 0)
    
    # Update CarROICrop device to match training device
    for pipeline_cfg in [cfg.train_dataloader.dataset.pipeline, 
                         cfg.val_dataloader.dataset.pipeline]:
        for transform in pipeline_cfg:
            if transform.get('type') == 'CarROICrop':
                transform['device'] = DEVICE
    
    # Configure WandB if enabled
    if USE_WANDB:
        print("\nConfiguring WandB...")
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
    
    # Enable resume if requested
    if RESUME:
        cfg.resume = True
        print("✓ Resume from checkpoint enabled")
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Build runner
    print("\n  Building runner...")
    runner = Runner.from_cfg(cfg)
    
    # Start training
    print("\n Starting training...")
    print("=" * 80)
    
    try:
        runner.train()
        
        print("\n" + "=" * 80)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved to: {WORK_DIR}")
        print(f"Checkpoints: {os.path.join(WORK_DIR, '*.pth')}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        print(f"Checkpoints saved to: {WORK_DIR}")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(" TRAINING FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    train()
