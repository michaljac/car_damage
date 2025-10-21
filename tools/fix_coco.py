# tools/fix_coco.py
import json, os
from PIL import Image
import yaml

def load_size(img_root, file_name):
    with Image.open(os.path.join(img_root, file_name)) as im:
        return im.size  # (W,H)

def fix_split(classes, in_ann, img_root, out_ann, cat_id_map, drop_empty=False):
    
    # Load the JSON and image info
    js = json.load(open(in_ann[0]))

    # build lookup map
    name2id = {c['name']: c['id'] for c in js['categories']}
    cat_id_map = {old_id: i+1 for i,(old_id,_) in enumerate(js['categories'])}

    for i,name in enumerate(classes): cat_id_map[name2id[name]] = i+1  # COCO ids usually start at 1
    id2img = {im['id']:im for im in js['images']}
    file2size = {im['file_name']: load_size(img_root, im['file_name']) for im in js['images']}

    out_anns, kept_img_ids = [], set()
    # iterate through all annotations
    for a in js['annotations']:
        im = id2img.get(a['image_id']); 
        if im is None or a['category_id'] not in cat_id_map: 
            continue
        
        # Clamp bounding boxes inside image borders
        W,H = file2size[im['file_name']]
        x,y,w,h = a['bbox']
        x2, y2 = x+w, y+h
        x, y = max(0,x), max(0,y)
        x2, y2 = min(W, x2), min(H, y2)
        w, h = x2-x, y2-y
        
        # drop invalid bboxes
        if w <= 1 or h <= 1: 
            continue
        
        # Recompute area & reassign category
        a['bbox'] = [float(x), float(y), float(w), float(h)]
        a['area'] = float(w*h)
        a['category_id'] = cat_id_map[a['category_id']]
        
        # Keep only valid annotations
        out_anns.append(a); kept_img_ids.add(a['image_id'])
        kept_img_ids.add(a['image_id'])

    # Optionally drop empty images
    if drop_empty:
        js['images'] = [im for im in js['images'] if im['id'] in kept_img_ids]

    # Save cleaned file
    js['categories'] = [{'id':i+1,'name':n} for i,n in enumerate(classes)]
    json.dump(js, open(out_ann,'w'))
    print(f"Saved {out_ann} | imgs={len(js['images'])} anns={len(out_anns)}")



if __name__ == '__main__':

    # Load yaml config file 
    with open("configs/preprocess.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cat_id_map = {}  # fill from categories in the file -> 0..5
    classes = cfg['classes']

    # should do it also for val and test
    in_ann = '/Data/annotations/annotations_val.json',
    img_root = '/Data/val2017'
    out_ann = '/Data/annotations/annotations_val_fixed.json'
    
    fix_split(classes, in_ann, img_root, out_ann, cat_id_map)
