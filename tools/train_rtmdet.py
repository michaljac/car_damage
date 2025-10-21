"""
Training script for RTMDet car damage detection model
"""
import os
import sys
import argparse
from mmdet.apis import train_detector
from mmdet.models import build_detector
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train RTMDet detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='working directory to save logs and models')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision training')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Override work_dir if specified
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    # Create work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        cfg.seed = args.seed
    
    # Enable resume if specified
    if args.resume:
        cfg.resume = True
    
    # Build runner
    runner = Runner.from_cfg(cfg)
    
    # Start training
    print(f"Starting training with config: {args.config}")
    print(f"Working directory: {cfg.work_dir}")
    print(f"Number of classes: {cfg.model.bbox_head.num_classes}")
    print(f"Max epochs: {cfg.train_cfg.max_epochs}")
    print("="*70)
    
    runner.train()
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Model checkpoints saved to: {cfg.work_dir}")


if __name__ == '__main__':
    main()

