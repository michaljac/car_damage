"""
Complete dataset preprocessing pipeline for COCO format annotations
Handles validation, moving problematic images, and creating clean datasets
"""
import json
import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
import yaml


def load_image_size(img_root, file_name):
    """Load image dimensions"""
    try:
        with Image.open(os.path.join(img_root, file_name)) as im:
            return im.size  # (W, H)
    except Exception as e:
        print(f"[ERROR] Cannot load image {file_name}: {e}")
        return None


def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Convert COCO bbox to YOLO format
    COCO: [x_min, y_min, width, height] (absolute pixels)
    YOLO: [x_center, y_center, width, height] (normalized 0-1)
    """
    x_min, y_min, bbox_width, bbox_height = coco_bbox
    
    # Calculate center point
    x_center = x_min + bbox_width / 2
    y_center = y_min + bbox_height / 2
    
    # Normalize to 0-1 range
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = bbox_width / img_width
    height_norm = bbox_height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def save_yolo_annotations(txt_path, annotations, img_width, img_height):
    """Save annotations in YOLO format"""
    try:
        with open(txt_path, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                
                # Convert to YOLO format
                x_center, y_center, width, height = coco_to_yolo_bbox(bbox, img_width, img_height)
                
                # Write: class_id x_center y_center width height
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save YOLO annotations to {txt_path}: {e}")
        return False


def move_to_raw(img_path, raw_dir, split_name, annotations=None, img_width=None, img_height=None):
    """
    Move problematic image to raw directory
    Also save its annotations in YOLO format for potential future use
    """
    if not os.path.exists(img_path):
        return False
    
    # Create raw directory structure
    raw_split_dir = os.path.join(raw_dir, f'{split_name}2017')
    os.makedirs(raw_split_dir, exist_ok=True)
    
    # Move image file
    filename = os.path.basename(img_path)
    dest_path = os.path.join(raw_split_dir, filename)
    
    try:
        shutil.move(img_path, dest_path)
        
        # Save annotations in YOLO format if provided
        if annotations and img_width and img_height and len(annotations) > 0:
            # Create .txt file with same name as image
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(raw_split_dir, f"{base_name}.txt")
            save_yolo_annotations(txt_path, annotations, img_width, img_height)
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to move {img_path}: {e}")
        return False


def preprocess_split(classes, in_ann, img_root, out_ann, split_name, data_root):
    """
    Preprocess a single split (train/val/test)
    - Move images with invalid categories to raw/
    - Move empty images to raw/
    - Remove orphan annotations
    - Fix out-of-bounds bboxes
    - Create clean annotation file
    """
    
    print(f"\n{'='*70}")
    print(f"[INFO] Processing {split_name} split")
    print(f"   Input annotation:  {in_ann}")
    print(f"   Image directory:   {img_root}")
    print(f"   Output annotation: {out_ann}")
    print(f"{'='*70}\n")
    
    # Load annotation file
    try:
        with open(in_ann, 'r') as f:
            coco = json.load(f)
    except Exception as e:
        print(f"[ERROR] Cannot load {in_ann}: {e}")
        return False
    
    # Build category map
    name2id = {c['name']: c['id'] for c in coco['categories']}
    valid_cat_ids = set()
    for i, name in enumerate(classes):
        if name in name2id:
            valid_cat_ids.add(name2id[name])
    
    print(f"Valid category IDs: {valid_cat_ids}")
    print(f"Total images in annotations: {len(coco['images'])}")
    print(f"Total annotations: {len(coco['annotations'])}")
    
    # Build lookup dicts
    id2img = {img['id']: img for img in coco['images']}
    
    # Track statistics
    stats = {
        'moved_invalid_category': [],
        'moved_empty_images': [],
        'removed_orphan_anns': 0,
        'fixed_out_of_bounds': 0,
        'removed_invalid_boxes': 0,
        'kept_images': set(),
        'kept_annotations': []
    }
    
    # Raw directory for problematic images
    raw_dir = os.path.join(data_root, 'raw')
    
    print(f"\nChecking all images and loading dimensions...")
    
    # Load all image sizes and check for issues
    file2size = {}
    images_to_remove = set()  # Image IDs to remove from final dataset
    
    for img in coco['images']:
        img_path = os.path.join(img_root, img['file_name'])
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            images_to_remove.add(img['id'])
            continue
        
        # Try to load image size
        size = load_image_size(img_root, img['file_name'])
        if size:
            file2size[img['file_name']] = size
            # Update image dimensions in annotation
            img['width'], img['height'] = size
        else:
            images_to_remove.add(img['id'])
    
    print(f"[INFO] Successfully loaded {len(file2size)} images")
    
    print(f"\n[PHASE 2] Processing annotations...")
    
    # Track which images have valid annotations
    images_with_valid_anns = set()
    
    # Track annotations for each image (to save with moved images)
    img_annotations = {}  # img_id -> list of annotations
    
    # Process all annotations
    for ann in coco['annotations']:
        img_id = ann['image_id']
        
        # Check for orphan annotations (image doesn't exist)
        if img_id not in id2img:
            stats['removed_orphan_anns'] += 1
            continue
        
        img = id2img[img_id]
        img_path = os.path.join(img_root, img['file_name'])
        
        # Track this annotation for the image
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
        
        # Skip if image was already marked for removal
        if img_id in images_to_remove:
            continue
        
        # Check for invalid category
        if ann['category_id'] not in valid_cat_ids:
            if img_id not in images_to_remove:
                print(f"[WARNING] Invalid category {ann['category_id']} for image: {img_path}")
                images_to_remove.add(img_id)
                if img_path not in [x[1] for x in stats['moved_invalid_category']]:
                    stats['moved_invalid_category'].append((img['file_name'], img_path, img_id))
            continue
        
        # Check if we have image size
        if img['file_name'] not in file2size:
            continue
        
        # Validate and fix bbox
        W, H = file2size[img['file_name']]
        bbox = ann['bbox']
        
        if len(bbox) != 4:
            stats['removed_invalid_boxes'] += 1
            continue
        
        x, y, w, h = bbox
        
        # Check and fix out-of-bounds
        if x < 0 or y < 0 or x + w > W or y + h > H:
            stats['fixed_out_of_bounds'] += 1
            # Clamp bbox
            x2, y2 = x + w, y + h
            x, y = max(0, x), max(0, y)
            x2, y2 = min(W, x2), min(H, y2)
            w, h = x2 - x, y2 - y
        
        # Check if bbox is still valid after clamping
        if w <= 1 or h <= 1:
            stats['removed_invalid_boxes'] += 1
            continue
        
        # Update bbox and area
        ann['bbox'] = [float(x), float(y), float(w), float(h)]
        ann['area'] = float(w * h)
        
        # Add iscrowd if missing
        if 'iscrowd' not in ann:
            ann['iscrowd'] = 0
        
        # Keep this annotation
        stats['kept_annotations'].append(ann)
        images_with_valid_anns.add(img_id)
    
    print(f"\n[PHASE 3] Identifying empty images...")
    
    # Find empty images (no valid annotations)
    for img in coco['images']:
        if img['id'] not in images_with_valid_anns and img['id'] not in images_to_remove:
            img_path = os.path.join(img_root, img['file_name'])
            if os.path.exists(img_path):
                print(f"[WARNING] Empty image (no annotations): {img_path}")
                images_to_remove.add(img['id'])
                stats['moved_empty_images'].append((img['file_name'], img_path, img['id']))
    
    print(f"\n[PHASE 4] Moving problematic images to raw/ directory...")
    print(f"           (Also saving annotations in YOLO format for future use)")
    
    moved_count = 0
    saved_txt_count = 0
    
    # Move images with invalid categories (save with YOLO annotations)
    for filename, img_path, img_id in stats['moved_invalid_category']:
        img = id2img[img_id]
        img_width = img.get('width')
        img_height = img.get('height')
        anns = img_annotations.get(img_id, [])
        
        if move_to_raw(img_path, raw_dir, split_name, anns, img_width, img_height):
            moved_count += 1
            if len(anns) > 0:
                saved_txt_count += 1
                print(f"   [MOVED] Invalid category: {filename} + annotations -> raw/{split_name}2017/")
            else:
                print(f"   [MOVED] Invalid category: {filename} -> raw/{split_name}2017/")
    
    # Move empty images (no annotations to save)
    for filename, img_path, img_id in stats['moved_empty_images']:
        if move_to_raw(img_path, raw_dir, split_name):
            moved_count += 1
            print(f"   [MOVED] Empty image: {filename} -> raw/{split_name}2017/")
    
    print(f"[INFO] Moved {moved_count} images to raw/ directory")
    print(f"[INFO] Saved {saved_txt_count} YOLO format annotation files")
    
    print(f"\n[PHASE 5] Creating clean annotation file...")
    
    # Keep only valid images
    clean_images = [img for img in coco['images'] if img['id'] not in images_to_remove]
    stats['kept_images'] = set(img['id'] for img in clean_images)
    
    # Remap category IDs to be sequential starting from 1
    cat_id_map = {}
    for i, name in enumerate(classes):
        if name in name2id:
            cat_id_map[name2id[name]] = i + 1
    
    # Update category IDs in annotations
    for ann in stats['kept_annotations']:
        if ann['category_id'] in cat_id_map:
            ann['category_id'] = cat_id_map[ann['category_id']]
    
    # Create clean COCO format
    clean_coco = {
        'images': clean_images,
        'annotations': stats['kept_annotations'],
        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(classes)]
    }
    
    # Save clean annotation file
    os.makedirs(os.path.dirname(out_ann), exist_ok=True)
    with open(out_ann, 'w') as f:
        json.dump(clean_coco, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"[SUMMARY] {split_name} split preprocessing complete")
    print(f"{'='*70}")
    print(f"Original:")
    print(f"   Images:      {len(coco['images'])}")
    print(f"   Annotations: {len(coco['annotations'])}")
    print(f"\nRemoved:")
    print(f"   Invalid category images:  {len(stats['moved_invalid_category'])}")
    print(f"   Empty images:             {len(stats['moved_empty_images'])}")
    print(f"   Orphan annotations:       {stats['removed_orphan_anns']}")
    print(f"   Invalid bounding boxes:   {stats['removed_invalid_boxes']}")
    print(f"\nFixed:")
    print(f"   Out-of-bounds boxes:      {stats['fixed_out_of_bounds']}")
    print(f"\nFinal clean dataset:")
    print(f"   Images:      {len(clean_images)}")
    print(f"   Annotations: {len(stats['kept_annotations'])}")
    print(f"\nOutput saved to: {out_ann}")
    print(f"{'='*70}\n")
    
    return True


def convert_clean_json_to_yolo(data_root, splits=['train', 'val', 'test']):
    """
    Convert clean JSON annotations to YOLO format text files
    Saves .txt files in the same directory as images:
    - data_root/train2017/*.txt (alongside images)
    - data_root/val2017/*.txt (alongside images)
    - data_root/test2017/*.txt (alongside images)
    
    Args:
        data_root: Root directory of dataset
        splits: List of splits to process
    
    Returns:
        True if successful, False otherwise
    """
    
    print(f"\n{'='*70}")
    print(f"[START] Converting Clean JSON Annotations to YOLO Format")
    print(f"{'='*70}")
    print(f"Data root: {data_root}")
    print(f"Splits to convert: {splits}")
    print(f"{'='*70}\n")
    
    total_stats = {
        'total_images': 0,
        'total_annotations': 0,
        'total_txt_files': 0
    }
    
    for split_name in splits:
        print(f"\n[INFO] Processing {split_name} split...")
        
        # Input paths
        clean_ann_file = os.path.join(data_root, 'annotations', f'annotations_{split_name}_clean.json')
        img_root = os.path.join(data_root, f'{split_name}2017')
        
        # Output label directory (same as image directory)
        label_dir = img_root
        
        # Check if clean annotation file exists
        if not os.path.exists(clean_ann_file):
            print(f"[WARNING] Clean annotation file not found: {clean_ann_file}")
            print(f"           Skipping {split_name} split")
            continue
        
        # Load clean annotations
        try:
            with open(clean_ann_file, 'r') as f:
                coco = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load {clean_ann_file}: {e}")
            continue
        
        # Label directory is the same as image directory, so no need to create it
        
        print(f"   Clean annotations: {clean_ann_file}")
        print(f"   Images: {len(coco['images'])}")
        print(f"   Annotations: {len(coco['annotations'])}")
        print(f"   Output directory: {label_dir}")
        
        # Build image ID to image info mapping
        id2img = {img['id']: img for img in coco['images']}
        
        # Build image ID to annotations mapping
        img2anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in img2anns:
                img2anns[img_id] = []
            img2anns[img_id].append(ann)
        
        # Convert annotations for each image
        txt_count = 0
        empty_count = 0
        error_count = 0
        
        for img in coco['images']:
            img_id = img['id']
            filename = img['file_name']
            img_width = img.get('width')
            img_height = img.get('height')
            
            # Get base filename without extension
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(label_dir, f"{base_name}.txt")
            
            # Get annotations for this image
            annotations = img2anns.get(img_id, [])
            
            if len(annotations) == 0:
                empty_count += 1
                # Create empty txt file
                try:
                    open(txt_path, 'w').close()
                    os.chmod(txt_path, 0o666)
                except Exception as e:
                    print(f"[ERROR] Failed to create empty file {txt_path}: {e}")
                    error_count += 1
                continue
            
            # Check if we have image dimensions
            if not img_width or not img_height:
                print(f"[WARNING] Image {filename} missing dimensions, skipping")
                error_count += 1
                continue
            
            # Save annotations in YOLO format
            if save_yolo_annotations(txt_path, annotations, img_width, img_height):
                os.chmod(txt_path, 0o666)
                txt_count += 1
            else:
                error_count += 1
        
        # Update total stats
        total_stats['total_images'] += len(coco['images'])
        total_stats['total_annotations'] += len(coco['annotations'])
        total_stats['total_txt_files'] += txt_count
        
        # Print split summary
        print(f"\n   [SUMMARY] {split_name} split:")
        print(f"      Created {txt_count} YOLO annotation files")
        print(f"      Empty images: {empty_count}")
        if error_count > 0:
            print(f"      Errors: {error_count}")
        print(f"      Output: {label_dir}/")
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"[COMPLETE] YOLO Format Conversion Finished")
    print(f"{'='*70}")
    print(f"Total images processed: {total_stats['total_images']}")
    print(f"Total annotations: {total_stats['total_annotations']}")
    print(f"Total YOLO .txt files created: {total_stats['total_txt_files']}")
    print(f"\nYOLO annotation files saved alongside images:")
    for split_name in splits:
        img_dir = os.path.join(data_root, f'{split_name}2017')
        if os.path.exists(img_dir):
            txt_files = len([f for f in os.listdir(img_dir) if f.endswith('.txt')])
            img_files = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   - {img_dir}/ ({img_files} images, {txt_files} .txt files)")
    
    print(f"\n{'='*70}")
    print(f"[NEXT STEP] Ready for YOLO training!")
    print(f"{'='*70}")
    print(f"Your dataset is now ready with:")
    print(f"   - Images + Labels together in: {data_root}/<split>2017/")
    print(f"\nUpdate your YOLO config with:")
    print(f"   path: {data_root}")
    print(f"   train: train2017")
    print(f"   val: val2017")
    print(f"   test: test2017")
    print(f"{'='*70}\n")
    
    return total_stats['total_txt_files'] > 0


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess COCO dataset: move problematic images and create clean annotations'
    )
    parser.add_argument('--config', default='configs/preprocess.yaml', 
                       help='config file path with class names')
    parser.add_argument('--data-root', default='/Data/coco', 
                       help='root directory of dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], 
                       help='splits to process')
    parser.add_argument('--convert-to-yolo', action='store_true',
                       help='convert clean annotations to YOLO format after preprocessing')
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        classes = cfg['classes']
    except Exception as e:
        print(f"[ERROR] Cannot load config {args.config}: {e}")
        return 1
    
    print(f"\n{'='*70}")
    print(f"[START] Dataset Preprocessing Pipeline")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Splits to process: {args.splits}")
    print(f"{'='*70}\n")
    
    # Count files before processing
    print(f"[INFO] Counting files before preprocessing...")
    before_counts = {}
    for split_name in args.splits:
        img_dir = os.path.join(args.data_root, f'{split_name}2017')
        ann_file = os.path.join(args.data_root, 'annotations', f'annotations_{split_name}.json')
        
        img_count = 0
        ann_count = 0
        
        if os.path.exists(img_dir):
            img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if os.path.exists(ann_file):
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    ann_count = len(data.get('annotations', []))
            except:
                pass
        
        before_counts[split_name] = {'images': img_count, 'annotations': ann_count}
        print(f"   {split_name}: {img_count} images, {ann_count} annotations")
    
    print(f"{'='*70}\n")
    
    # Process each split
    success_count = 0
    after_counts = {}
    
    for split_name in args.splits:
        # Input paths
        in_ann = os.path.join(args.data_root, 'annotations', f'annotations_{split_name}.json')
        img_root = os.path.join(args.data_root, f'{split_name}2017')
        
        # Output paths
        out_ann = os.path.join(args.data_root, 'annotations', f'annotations_{split_name}_clean.json')
        
        # Check if input exists
        if not os.path.exists(in_ann):
            print(f"[WARNING] Skipping {split_name}: {in_ann} does not exist\n")
            continue
        
        if not os.path.exists(img_root):
            print(f"[WARNING] Skipping {split_name}: {img_root} does not exist\n")
            continue
        
        # Process split
        try:
            if preprocess_split(classes, in_ann, img_root, out_ann, split_name, args.data_root):
                success_count += 1
                
                # Count files after processing
                clean_img_count = 0
                clean_ann_count = 0
                
                if os.path.exists(img_root):
                    clean_img_count = len([f for f in os.listdir(img_root) if f.endswith(('.jpg', '.jpeg', '.png'))])
                
                if os.path.exists(out_ann):
                    try:
                        with open(out_ann, 'r') as f:
                            data = json.load(f)
                            clean_ann_count = len(data.get('annotations', []))
                    except:
                        pass
                
                after_counts[split_name] = {'images': clean_img_count, 'annotations': clean_ann_count}
                
        except Exception as e:
            print(f"[ERROR] Failed to process {split_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary with before/after comparison
    print(f"\n{'='*70}")
    print(f"[COMPLETE] Preprocessing Pipeline Finished")
    print(f"{'='*70}")
    print(f"Successfully processed: {success_count}/{len(args.splits)} splits")
    
    # Before/After comparison table
    print(f"\n{'='*70}")
    print(f"BEFORE vs AFTER Comparison:")
    print(f"{'='*70}")
    
    total_before_imgs = 0
    total_before_anns = 0
    total_after_imgs = 0
    total_after_anns = 0
    total_moved = 0
    
    for split_name in args.splits:
        if split_name in before_counts:
            before = before_counts[split_name]
            after = after_counts.get(split_name, {'images': 0, 'annotations': 0})
            
            moved = before['images'] - after['images']
            
            print(f"\n{split_name.upper()}:")
            print(f"   Images:      {before['images']:6d} -> {after['images']:6d}  (moved: {moved})")
            print(f"   Annotations: {before['annotations']:6d} -> {after['annotations']:6d}  (removed: {before['annotations'] - after['annotations']})")
            
            total_before_imgs += before['images']
            total_before_anns += before['annotations']
            total_after_imgs += after['images']
            total_after_anns += after['annotations']
            total_moved += moved
    
    print(f"\n{'-'*70}")
    print(f"TOTAL:")
    print(f"   Images:      {total_before_imgs:6d} -> {total_after_imgs:6d}  (moved: {total_moved})")
    print(f"   Annotations: {total_before_anns:6d} -> {total_after_anns:6d}  (removed: {total_before_anns - total_after_anns})")
    print(f"{'='*70}")
    
    print(f"\nClean annotation files created:")
    for split_name in args.splits:
        out_ann = os.path.join(args.data_root, 'annotations', f'annotations_{split_name}_clean.json')
        if os.path.exists(out_ann):
            print(f"   - {out_ann}")
    
    raw_dir = os.path.join(args.data_root, 'raw')
    if os.path.exists(raw_dir):
        print(f"\nProblematic images moved to:")
        print(f"   - {raw_dir}/")
        for split_name in args.splits:
            raw_split = os.path.join(raw_dir, f'{split_name}2017')
            if os.path.exists(raw_split):
                files = os.listdir(raw_split)
                img_count = len([f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))])
                txt_count = len([f for f in files if f.endswith('.txt')])
                print(f"     - {split_name}2017/ ({img_count} images, {txt_count} YOLO annotation files)")
    
    print(f"\n Update your training config to use the _clean.json files:")
    print(f"   train_ann_file = 'annotations/annotations_train_clean.json'")
    print(f"   val_ann_file = 'annotations/annotations_val_clean.json'")
    print(f"   test_ann_file = 'annotations/annotations_test_clean.json'")
    print(f"{'='*70}\n")
    
    # Convert to YOLO format if requested
    if args.convert_to_yolo and success_count > 0:
        if convert_clean_json_to_yolo(args.data_root, args.splits):
            print(f"[SUCCESS] Dataset is ready for YOLO training!\n")
        else:
            print(f"[WARNING] YOLO conversion completed with errors\n")
    
    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    exit(main())
