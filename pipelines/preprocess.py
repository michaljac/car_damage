import os
import json

def process_annotations_without_images(images, annotations):
    image_ids = {img['id'] for img in images}
    for ann in annotations:
        if ann['image_id'] not in image_ids:
            print(f"Annotation with ID {ann['id']} has no corresponding image.")
            
def process_images_without_annotations(images, image_id_to_annotations):
    for img in images:
        img_id = img['id']
        anns = image_id_to_annotations.get(img_id, [])
        if not anns:
            print(f"Image ID {img_id} has no annotations.")

def process_annotations_with_invalid_category(annotations, valid_categories):
    for ann in annotations:
        if ann['category_id'] not in valid_categories:
            print(f"Annotation with ID {ann['id']} has an invalid category ID {ann['category_id']}.")

def process_annotations_with_invalid_bbox(images, annotations):
    image_id_to_image = {img['id']: img for img in images}
    for ann in annotations:
        img = image_id_to_image.get(ann['image_id'])
        if img:
            img_width, img_height = img['width'], img['height']
            x, y, w, h = ann['bbox']
            if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                print(f"Annotation with ID {ann['id']} has an invalid bbox {ann['bbox']} for image ID {img['id']}.")

def process_duplicate_images(images):
    seen = set()
    for img in images:
        img_file_name = img['file_name']
        if img_file_name in seen:
            print(f"Duplicate image found: {img_file_name}")
        else:
            seen.add(img_file_name)

def process_duplicate_annotations(annotations):
    seen = set()
    for ann in annotations:
        ann_key = (ann['image_id'], tuple(ann['bbox']), ann['category_id'])
        if ann_key in seen:
            print(f"Duplicate annotation found: {ann}")
        else:
            seen.add(ann_key)

def process_missing_image_files(images, data_dir):
    """Check if image files referenced in JSON actually exist on disk"""
    missing_files = []
    for img in images:
        img_path = os.path.join(data_dir, img['file_name'])
        if not os.path.exists(img_path):
            print(f"Image file missing from disk: {img['file_name']} (ID: {img['id']})")
            missing_files.append(img['id'])
    return missing_files

def process_unlisted_image_files(images, data_dir):
    """Find image files on disk that aren't listed in the JSON annotations"""
    # Get all image files from directory
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(file)
    
    # Get listed image filenames from JSON
    json_image_files = {img['file_name'] for img in images}
    
    # Find files that exist on disk but not in JSON
    unlisted_files = []
    for img_file in image_files:
        if img_file not in json_image_files:
            print(f"Image file exists on disk but not in annotations: {img_file}")
            unlisted_files.append(img_file)
    return unlisted_files



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


    # process_annotations_without_images(images, annotations)
    # process_images_without_annotations(images, image_id_to_annotations)
    # process_annotations_with_invalid_category(annotations, valid_categories)
    process_annotations_with_invalid_bbox(images, annotations)
    # process_duplicate_images(images)
    # process_duplicate_annotations(annotations)
    # process_missing_image_files(images, data_dir)
    # process_unlisted_image_files(images, data_dir)
    
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