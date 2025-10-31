import os
import json
import shutil
import yaml
import cv2
from mmdet.apis import DetInferencer
# Register all modules to avoid the mmdet scope warning
import torch
from mmdet.utils import register_all_modules

def visualize_detection(img_path, box, score, output_dir, filename):
    """Visualize vehicle detection and save the result"""
    
    # Read and copy image
    img_array = cv2.imread(img_path)
    if img_array is None:
        print(f"Failed to load image: {img_path}")
        return False
        
    img_vis = img_array.copy()
    
    # Draw bounding box
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Add score text
    cv2.putText(img_vis, f"Score: {score:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"vis_{filename}")
    cv2.imwrite(output_path, img_vis)


def get_user_confirmation(message):
    """Helper function to get user confirmation"""
    while True:
        response = input(f"{message} (y/n): ").lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please answer 'y' or 'n'")

def process_annotations_without_images(images, annotations, data):
    """Process and optionally remove annotations without corresponding images"""
    image_ids = {img['id'] for img in images}
    invalid_anns = [ann for ann in annotations if ann['image_id'] not in image_ids]
    
    if invalid_anns:
        print(f"\nFound {len(invalid_anns)} annotations without corresponding images")
        if get_user_confirmation("Would you like to remove these annotations?"):
            data['annotations'] = [ann for ann in annotations if ann['image_id'] in image_ids]
            print(f"Removed {len(invalid_anns)} invalid annotations")
            return True
    return False
            
def process_images_without_annotations(images, image_id_to_annotations, data, data_dir):
    """Process and optionally remove images without annotations"""
    invalid_images = [img for img in images if not image_id_to_annotations.get(img['id'])]
    
    if invalid_images:
        print(f"\nFound {len(invalid_images)} images without annotations")
        if get_user_confirmation("Would you like to remove these images from JSON and disk?"):
            # Remove from JSON
            data['images'] = [img for img in images if image_id_to_annotations.get(img['id'])]
            # Remove from disk
            for img in invalid_images:
                img_path = os.path.join(data_dir, img['file_name'])
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"Removed file: {img_path}")
            print(f"\nRemoved {len(invalid_images)} images from JSON and disk")
            return True
    return False

def process_annotations_with_invalid_category(annotations, valid_categories, data):
    """Process and optionally remove annotations with invalid categories"""
    invalid_anns = [ann for ann in annotations if ann['category_id'] not in valid_categories]
    
    if invalid_anns:
        print(f"\nFound {len(invalid_anns)} annotations with invalid categories")
        if get_user_confirmation("Would you like to remove these annotations?"):
            data['annotations'] = [ann for ann in annotations if ann['category_id'] in valid_categories]
            print(f"\nRemoved {len(invalid_anns)} invalid annotations")
            return True
    return False

def process_annotations_with_invalid_bbox(images, annotations, data):
    """Process and optionally remove annotations with invalid bboxes"""
    image_id_to_image = {img['id']: img for img in images}
    invalid_anns = []
    
    for ann in annotations:
        img = image_id_to_image.get(ann['image_id'])
        if img:
            img_width, img_height = img['width'], img['height']
            x, y, w, h = ann['bbox']
            if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                invalid_anns.append(ann)
    
    if invalid_anns:
        print(f"\nFound {len(invalid_anns)} annotations with invalid bboxes")
        if get_user_confirmation("Would you like to remove these annotations?"):
            data['annotations'] = [ann for ann in annotations if ann not in invalid_anns]
            print(f"\nRemoved {len(invalid_anns)} invalid annotations")
            return True
    return False

def process_duplicate_images(images, data, data_dir):
    """Process and optionally remove duplicate images"""
    seen = {}
    duplicates = []
    
    for img in images:
        if img['file_name'] in seen:
            duplicates.append(img)
        else:
            seen[img['file_name']] = img
    
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate images")
        if get_user_confirmation("Would you like to remove duplicate images from JSON and disk?"):
            data['images'] = list(seen.values())
            # Remove duplicate files from disk
            for img in duplicates:
                img_path = os.path.join(data_dir, img['file_name'])
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"Removed duplicate file: {img_path}")
            print(f"\nRemoved {len(duplicates)} duplicate images")
            return True
    return False

def process_duplicate_annotations(annotations, data):
    """Process and optionally remove duplicate annotations"""
    seen = set()
    duplicates = []
    unique_anns = []
    
    for ann in annotations:
        ann_key = (ann['image_id'], tuple(ann['bbox']), ann['category_id'])
        if ann_key in seen:
            duplicates.append(ann)
        else:
            seen.add(ann_key)
            unique_anns.append(ann)
    
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate annotations")
        if get_user_confirmation("Would you like to remove duplicate annotations?"):
            data['annotations'] = unique_anns
            print(f"\nRemoved {len(duplicates)} duplicate annotations")
            return True
    return False

def process_missing_image_files(images, data_dir, data):
    """Process and optionally remove entries for missing image files"""
    missing_images = []
    for img in images:
        img_path = os.path.join(data_dir, img['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img)
    
    if missing_images:
        print(f"\nFound {len(missing_images)} images missing from disk")
        if get_user_confirmation("Would you like to remove these entries from JSON?"):
            data['images'] = [img for img in images if img not in missing_images]
            # Also remove corresponding annotations
            missing_ids = {img['id'] for img in missing_images}
            data['annotations'] = [ann for ann in data['annotations'] 
                                 if ann['image_id'] not in missing_ids]
            print(f"\nRemoved {len(missing_images)} image entries and their annotations")
            return True
    return False

def process_unlisted_image_files(images, data_dir):
    """Process and optionally remove unlisted image files from disk"""
    json_image_files = {img['file_name'] for img in images}
    unlisted_files = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if file not in json_image_files:
                    unlisted_files.append(os.path.join(root, file))
    
    if unlisted_files:
        print(f"\nFound {len(unlisted_files)} images on disk that aren't in JSON")
        if get_user_confirmation("Would you like to remove these files from disk?"):
            for file_path in unlisted_files:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            return True
    return False


def process_ann_file(ann_file, data_dir, valid_categories):
    with open(ann_file, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])

    image_id_to_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)

    # Get initial counts for verification
    initial_img_count = len(images)
    initial_ann_count = len(annotations)

    # clean the dataset
    processing_steps = [
        (process_annotations_without_images, [images, annotations, data]),
        (process_images_without_annotations, [images, image_id_to_annotations, data, data_dir]),
        (process_annotations_with_invalid_category, [annotations, valid_categories, data]),
        (process_annotations_with_invalid_bbox, [images, annotations, data]),
        (process_duplicate_images, [images, data, data_dir]),
        (process_duplicate_annotations, [annotations, data]),
        (process_missing_image_files, [images, data_dir, data]),
        (process_unlisted_image_files, [images, data_dir])
    ]

    # Run all processing steps but only set changes_made to True for the first successful change
    changes_made = False
    for func, args in processing_steps:
        if func(*args):
            changes_made = True
            # Update our working copies after each change
            images = data['images']
            annotations = data['annotations']
            
    # Print change summary
    if changes_made:
        print("\nChanges summary:")
        print(f"Images: {initial_img_count} -> {len(images)}")
        print(f"Annotations: {initial_ann_count} -> {len(annotations)}")
        
        if get_user_confirmation("Would you like to save the changes to the JSON file?"):
            # Create backup
            backup_dir = os.path.join(os.path.dirname(ann_file), 'backup')
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"{os.path.basename(ann_file)}.bak")
            shutil.copy2(ann_file, backup_path)
            
            # # Save changes
            # with open(ann_file, 'w') as f:
            #     json.dump(data, f, indent=2)
            # print(f"Saved updated annotations to: {ann_file}")

            # Save changes to temp file first
            filename, ext = os.path.splitext(ann_file)
            temp_file = filename + '_tmp' + ext
            try:
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                # Atomic rename operation
                os.replace(temp_file, ann_file)
                print(f"Saved updated annotations to: {ann_file}")
            except Exception as e:
                # Clean up temp file if something goes wrong
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise e  # Re-raise the exception after cleanup
    
    else:
        print("Annotation file is ready for training.")



def add_car_category(cfg, ann_file, image_dir, car_category_id=7):
    
    # register mmdet modules
    register_all_modules()

    # Model config and checkpoint paths
    vehicle_classes = cfg['vehicle_classes']
    config_file = cfg['config_file'] 
    checkpoint_file = cfg['checkpoint_file']

    # Download checkpoint if not exists
    if not os.path.exists(checkpoint_file):
        checkpoint_dir = os.path.dirname(checkpoint_file)
        os.makedirs(checkpoint_dir, exist_ok=True)
        url = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        print(f"Downloading checkpoint from {url}")
        torch.hub.download_url_to_file(url, checkpoint_file)

    # Initialize inferencer
    inferencer = DetInferencer(
        config_file,
        checkpoint_file,
        device='cpu'
    )

    with open(ann_file, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    image_id_to_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)

    for img in images:
        img_path = os.path.join(image_dir, img['file_name'])
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            continue
        
        results = inferencer(img_path)
        result = results['predictions'][0] 
        
        # Filter for vehicle classes only
        vehicle_indices = [
            i for i, label in enumerate(result['labels']) 
            if label in vehicle_classes.values()
        ] 

        # Get all vehicle detections with score >= score_threshold
        valid_detections = []
        for idx in vehicle_indices:
            score = result['scores'][idx]
            if score >= cfg['score_threshold']:
                box = result['bboxes'][idx]
                area = (box[2] - box[0]) * (box[3] - box[1])
                valid_detections.append({
                    'box': box,
                    'score': score,
                    'area': area,
                    'type': 'vehicle'  # Single class for all vehicles
                })

        # Keep only the largest vehicle detection
        if valid_detections:
            largest_vehicle = max(valid_detections, key=lambda x: x['area'])
            box = largest_vehicle['box']
            
            # Convert from [x1,y1,x2,y2] to [x,y,w,h] format
            x = float(box[0])
            y = float(box[1])
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            
            new_ann = {
                'id': len(annotations) + 1,
                'image_id': img['id'],
                'category_id': car_category_id,  # Single category ID for all vehicles
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'score': float(largest_vehicle['score'])
            }
            annotations.append(new_ann)
            print(f"Added vehicle annotation for image ID {img['id']} "
                  f"with bbox {new_ann['bbox']} (score: {largest_vehicle['score']:.2f})")


        # visualize the detection results
        visualize = get_user_confirmation("Would you like to visualize and save the detections?")
        if visualize:
            vis_dir = cfg['vis_dir']
            visualize_detection(
                img_path=img_path,
                box=box,
                score=score,
                output_dir=vis_dir,
                filename=img['file_name']
            )

    # Save updated annotations back to file

    
if __name__ == '__main__':
    

    with open('configs/config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


    valid_categories = cfg['valid_categories']
    data_dir = cfg['data_dir'] 
    annotations_dir = cfg['annotations_dir']
    annotations_train = cfg['annotations_train']
    annotations_val = cfg['annotations_val']
    annotations_test = cfg['annotations_test']

    # process each annotation file
    # print("\nChecking if data cleaning is needed...")
    annotations_files = [annotations_train, annotations_val, annotations_test]
    # for ann_file in annotations_files:
    #     if not os.path.exists(ann_file):
    #         raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    #     else:
    #         print(f"\nProcessing annotation file: {ann_file}")
    #         ann_file_basename = os.path.basename(ann_file)
    #         split = ann_file_basename.replace("annotations_", "").replace(".json", "")
    #         # Construct image directory path
    #         image_dir = os.path.join(data_dir, f"{split}2017")
    #         process_ann_file(ann_file, image_dir, valid_categories)
    

    # inference each image and save the car ROIs to json file
    print("\nAll annotation files processed.")
    print("Now let's do inference cars from images.")
    for ann_file in annotations_files:
        print(f"\nProcessing annotation file: {ann_file}")
        ann_file_basename = os.path.basename(ann_file)
        split = ann_file_basename.replace("annotations_", "").replace(".json", "")
        # Construct image directory path
        image_dir = os.path.join(data_dir, f"{split}2017")
        add_car_category(cfg, ann_file, image_dir)