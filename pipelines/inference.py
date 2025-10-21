"""
Inference script for car damage detection
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--input', required=True, help='input image file or directory')
    parser.add_argument('--output', default='./inference_results', help='output directory')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold')
    parser.add_argument('--device', default='cuda:0', help='device to use')
    parser.add_argument('--show', action='store_true', help='show results in window')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize model
    print(f"Loading model from {args.checkpoint}...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print(f"[SUCCESS] Model loaded successfully!")
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
    else:
        raise ValueError(f"Input path {args.input} does not exist")
    
    print(f"\n[INFO] Found {len(image_files)} images to process")
    print(f"[INFO] Score threshold: {args.score_thr}")
    print(f"[INFO] Output directory: {args.output}")
    print("="*70)
    
    # Initialize visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        img_name = img_path.name
        print(f"\n[{i}/{len(image_files)}] Processing: {img_name}")
        
        # Run inference
        result = inference_detector(model, str(img_path))
        
        # Count detections per class
        pred_instances = result.pred_instances
        pred_instances = pred_instances[pred_instances.scores > args.score_thr]
        
        if len(pred_instances) > 0:
            print(f"   [INFO] Found {len(pred_instances)} detections:")
            
            # Count by class
            class_counts = {}
            for label in pred_instances.labels:
                class_name = model.dataset_meta['classes'][label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                print(f"      - {class_name}: {count}")
        else:
            print(f"   [INFO] No detections above threshold")
        
        # Visualize
        img = cv2.imread(str(img_path))
        visualizer.add_datasample(
            img_name,
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=0 if args.show else 0,
            pred_score_thr=args.score_thr
        )
        
        # Save result
        output_path = os.path.join(args.output, f'pred_{img_name}')
        vis_img = visualizer.get_image()
        cv2.imwrite(output_path, vis_img)
        print(f"   [SAVED] Saved to: {output_path}")
    
    print("\n" + "="*70)
    print(f"[COMPLETE] Inference complete! Results saved to {args.output}")


if __name__ == '__main__':
    main()

