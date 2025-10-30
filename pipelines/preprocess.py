import os
import json
import shutil

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
        print(f"Found {len(invalid_anns)} annotations without corresponding images")
        if get_user_confirmation("Would you like to remove these annotations?"):
            data['annotations'] = [ann for ann in annotations if ann['image_id'] in image_ids]
            print(f"Removed {len(invalid_anns)} invalid annotations")
            return True
    return False
            
def process_images_without_annotations(images, image_id_to_annotations, data, data_dir):
    """Process and optionally remove images without annotations"""
    invalid_images = [img for img in images if not image_id_to_annotations.get(img['id'])]
    
    if invalid_images:
        print(f"Found {len(invalid_images)} images without annotations")
        if get_user_confirmation("Would you like to remove these images from JSON and disk?"):
            # Remove from JSON
            data['images'] = [img for img in images if image_id_to_annotations.get(img['id'])]
            # Remove from disk
            for img in invalid_images:
                img_path = os.path.join(data_dir, img['file_name'])
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"Removed file: {img_path}")
            print(f"Removed {len(invalid_images)} images from JSON and disk")
            return True
    return False

def process_annotations_with_invalid_category(annotations, valid_categories, data):
    """Process and optionally remove annotations with invalid categories"""
    invalid_anns = [ann for ann in annotations if ann['category_id'] not in valid_categories]
    
    if invalid_anns:
        print(f"Found {len(invalid_anns)} annotations with invalid categories")
        if get_user_confirmation("Would you like to remove these annotations?"):
            data['annotations'] = [ann for ann in annotations if ann['category_id'] in valid_categories]
            print(f"Removed {len(invalid_anns)} invalid annotations")
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
        print(f"Found {len(invalid_anns)} annotations with invalid bboxes")
        if get_user_confirmation("Would you like to remove these annotations?"):
            data['annotations'] = [ann for ann in annotations if ann not in invalid_anns]
            print(f"Removed {len(invalid_anns)} invalid annotations")
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
        print(f"Found {len(duplicates)} duplicate images")
        if get_user_confirmation("Would you like to remove duplicate images from JSON and disk?"):
            data['images'] = list(seen.values())
            # Remove duplicate files from disk
            for img in duplicates:
                img_path = os.path.join(data_dir, img['file_name'])
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"Removed duplicate file: {img_path}")
            print(f"Removed {len(duplicates)} duplicate images")
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
        print(f"Found {len(duplicates)} duplicate annotations")
        if get_user_confirmation("Would you like to remove duplicate annotations?"):
            data['annotations'] = unique_anns
            print(f"Removed {len(duplicates)} duplicate annotations")
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
        print(f"Found {len(missing_images)} images missing from disk")
        if get_user_confirmation("Would you like to remove these entries from JSON?"):
            data['images'] = [img for img in images if img not in missing_images]
            # Also remove corresponding annotations
            missing_ids = {img['id'] for img in missing_images}
            data['annotations'] = [ann for ann in data['annotations'] 
                                 if ann['image_id'] not in missing_ids]
            print(f"Removed {len(missing_images)} image entries and their annotations")
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
        print(f"Found {len(unlisted_files)} images on disk that aren't in JSON")
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

    processing_steps = [
        # (process_annotations_without_images, [images, annotations, data]),
        # (process_images_without_annotations, [images, image_id_to_annotations, data, data_dir]),
        (process_annotations_with_invalid_category, [annotations, valid_categories, data]),
        # (process_annotations_with_invalid_bbox, [images, annotations, data]),
        # (process_duplicate_images, [images, data, data_dir]),
        # (process_duplicate_annotations, [annotations, data]),
        # (process_missing_image_files, [images, data_dir, data]),
        # (process_unlisted_image_files, [images, data_dir])
    ]

    # Run all processing steps but only set changes_made to True for the first successful change
    changes_made = False
    for func, args in processing_steps:
        result = func(*args)
        if not changes_made:
            changes_made = result
            
    # Save changes if any were made
    if changes_made:
        if get_user_confirmation("Would you like to save the changes to the JSON file?"):
            backup_path = f"{ann_file}.backup"
            shutil.copy2(ann_file, backup_path)
            print(f"Created backup at: {backup_path}")
            with open(ann_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved updated annotations to: {ann_file}")
    
if __name__ == '__main__':
    
    valid_categories = range(1, 7)
    data_dir = "/Data/coco"
    annotations_dir = "/Data/coco/annotations"
    annotations_train = os.path.join(annotations_dir, "annotations_train.json")
    annotations_val = os.path.join(annotations_dir, "annotations_val.json")
    annotations_test = os.path.join(annotations_dir, "annotations_test.json")

    annotations_files = [annotations_train, annotations_val, annotations_test]
    for ann_file in annotations_files:
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        else:
            print(f"\nProcessing annotation file: {ann_file}")
            ann_file_basename = os.path.basename(ann_file)
            split = ann_file_basename.replace("annotations_", "").replace(".json", "")
            # Construct image directory path
            image_dir = os.path.join(data_dir, f"{split}2017")
            process_ann_file(ann_file, image_dir, valid_categories)