#!/usr/bin/env python3
"""
Standalone script to convert clean JSON annotations to YOLO format
This script can be run independently after preprocessing is complete
"""
import argparse
from preprocess import convert_clean_json_to_yolo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert clean COCO JSON annotations to YOLO format'
    )
    parser.add_argument('--data-root', default='/Data/coco',
                       help='root directory of dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='splits to convert')
    
    args = parser.parse_args()
    
    # Convert to YOLO format
    success = convert_clean_json_to_yolo(args.data_root, args.splits)
    
    exit(0 if success else 1)

