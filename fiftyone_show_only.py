import glob
import os, shutil
import yaml
from utils_dir.utils_fiftyone import read_label_file, night_or_day, export_view_to_txt, get_files_list, sort_file_paths
import imagesize 
import cv2

# important: set custom paths for fiftyone config files config
os.environ['FIFTYONE_CONFIG_PATH'] = 'config/config.json'
os.environ['FIFTYONE_APP_CONFIG_PATH'] = 'config/app_config.json'

# important: Bug fix for importing `fiftyone`, based on: https://github.com/voxel51/fiftyone/issues/1334
folder = "/root/.fiftyone"

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.brain as fob

def save_files_list(text_file_path, view):
    with open(text_file_path, 'a+') as file:
        for sample in view:
            file.write(sample.filepath + '\n')

def check_list(img_files):
    """
    Validates images paths in list, checks for matching labels and converts xml to txt if required
    """
    convert_xml_to_txt=False
    verified_list = []
    isverified = True
    for img_file in img_files:
        
        # get txt path based on the img file path (assuming located in the same folder)        
        parent_dir = os.path.abspath(os.path.join(img_file, os.pardir))
        basename = os.path.basename(img_file)
        filename, file_ext = os.path.splitext(basename)
        xml_path = os.path.join(parent_dir, filename + '.xml')
        label_path = os.path.join(parent_dir, filename + '.txt')
        
        # check if xml file exists
        if not os.path.exists(label_path):
            
            # check if its a background image (without label file)
            if 'background' not in [p.lower() for p in label_path.split('/')]:
                print(f"skipping {label_path} label file does not exist!")
            continue
        
        else:
            
            with open(label_path, 'r') as file:
                for line in file.read().strip().splitlines():
                    line = line.strip().split(" ")

                    if len(line) == 6:
                        isverified = False
                        break
                
        if isverified:
            verified_list.append(img_file)
        
    return verified_list


def run_fiftyone(img_files, cfg=None):
    
    include_background = cfg['include_background']
    
    # Create custom samples for the dataset
    samples = []
    export_list = []
    label_files_count = 0
    for img_file in img_files:
        image_w, image_h = imagesize.get(img_file)
        if min(image_w, image_h) < 1:
            print('corrupted file: ', img_file)
            continue
        
        img_size = (image_w, image_h)
        is_background = False
        if not os.path.exists(img_file):
            print(f"{img_file} image file does not exist!")
            continue
        
        # get txt path based on the img file path (assuming located in the same folder) 
        parent_dir = os.path.abspath(os.path.join(img_file, os.pardir))
        basename = os.path.basename(img_file)
        filename, file_ext = os.path.splitext(basename)
        label_path = ''
        if img_file.endswith('jpg'):
            label_path = img_file[:-4] + '.txt'
        elif img_file.endswith('png'):
            label_path = img_file[:-4] + '.txt'
        elif img_file.endswith('jpeg'):
            label_path = img_file[:-5] + '.txt'
        
        # check if txt file exists
        if not os.path.exists(label_path):
            label_path = None

            if cfg['with_labels']:
                print(f"{label_path} label txt does not exist!")
                continue
        else:
            label_files_count+=1
            
        sample = fo.Sample(filepath=img_file)
                
        detections = []          
        detections_area = []          
        if (label_path is not None) and (os.path.exists(label_path)):
            # classes_dict = {"person":'0', "man":'0', "men":'0', "car":'1', "armed_person":'2', "armed":'2',"motorcycle":'3', "drone":'4' }
            # fiftyone_classes_dict = {'0':"person",'1':"vehicle"}
            classes_dict = cfg['classes_dict']
            fiftyone_classes_dict = cfg['fiftyone_classes_dict']
            label_list, bounding_box_list = read_label_file(label_path, img_size)
            # label_list, bounding_box_list = read_label_file(label_path, img_size, classes_dict, fiftyone_classes_dict)
            
            if len(label_list) > 0:
                for i in range(len(label_list)):
                    detections.append(fo.Detection(label=label_list[i], bounding_box=bounding_box_list[i]))
                    detections_area.append((bounding_box_list[i][-1]*image_w) * (bounding_box_list[i][-2]*image_h)) 
            else:
                if include_background:
                    is_background = True
                else:
                    continue
                
            sample["ground_truth"] = fo.Detections(detections=detections) 
            
        # get image shape
        w, h = imagesize.get(img_file)
        sample["resolution"] = str(w) + ' x ' + str(h)
        sample["background"] = str(is_background)
        sample["sub_dir"] = parent_dir.split('/')[-1]
        sample["group_dir"] = parent_dir.split('/')[-2]
        sample["objects"] = len(detections)
        sample["objects_size"] = sum(detections_area)
        sample["objects_relative_size"] = (sum(detections_area)) / (image_w * image_h)
        
        cam_type = 'all'
        cfg['night_keywords'] = ['night', 'sparrow', 'lwir', 'flir', 'thermal', 'inverted']
        cfg['auto_detect_night'] = False
        if cam_type != 'all':
            sample["cam_type"] = cam_type
        else:
            sample["cam_type"] = night_or_day(img_file, cfg['night_keywords'])
        
        # for location in locations:
        #     if location in img_file.lower():
        #         sample["location"] = location  
        
        samples.append(sample)
        # export_list.append(img_file)
                
    print(f'\n*** Total {len(img_files)} images and {label_files_count} txt label files')
    
    # Create dataset
    dataset = fo.Dataset("my-custom-detection-dataset")
    dataset.add_samples(samples)

    # sort by filepath
    # dataset = dataset.sort_by("filepath")
    
    
    batch_size = 4000 # cfg['batch_size'] 
    # Dividing the list of files into batches
    for i in range(0, len(samples), batch_size):
        batch = dataset[i:i + batch_size]

        print("batch size:", i, i+len(batch))
        session = fo.launch_app(batch) # batch.shuffle()

        user_input = input('compute uniqueness?  ')
        if user_input.lower() == 'y':
            
            fob.compute_uniqueness(dataset, num_workers=4)
            
            # Sort in increasing order of uniqueness (least unique first)
            dups_view = dataset.sort_by("uniqueness", reverse=True)
            session.view = dups_view
        
        # Perform similarity search
        user_input = input('compute similarity?  ')
        if user_input.lower() == 'y':

            fob.compute_similarity(
                                    dataset,
                                    # model="clip-vit-base32-torch",
                                    model="mobilenet-v2-imagenet-torch",
                                    brain_key="img_sim",
                                )
            
            # Choose a random image from the dataset
            # query_id = dataset.take(1).first().id
            query_id = input('Enter base image query id:  ')
            # query_id = '669f56fc9327a2d87055bfe4'

            kn = input('Enter Number of similar images to present:  ')
            if kn.isnumeric():
                kn = int(kn)
            else:
                kn = 100

            # Programmatically construct a view containing the 15 most similar images
            similarity_view = dataset.sort_by_similarity(query_id, k=kn, brain_key="img_sim")
            session.view = similarity_view

            # Compute embeddings using a pre-trained model (ResNet-50)
            # import fiftyone.zoo as foz
            # model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
            # embeddings = dataset.compute_embeddings(model)
            # # You can specify the number of clusters if you want to group images
            # fob.compute_similarity(dataset, brain_key="image_similarity", model="resnet50-imagenet-torch")
            # # Sort the dataset by similarity to a reference image
            # reference_image = dataset.first()  # Example: using the first image as reference
            # similarity_view = dataset.sort_by_similarity(reference_image, k=len(dataset))

            # # Visualize the sorted dataset in the FiftyOne app
            # session.view = similarity_view


        print('to create a files list from selected tagged imags enter "y" ')
        print('or press Enter to contiue')
        user_input = input('Input: ')
        if user_input.lower() == 'y':
            data_selection(dataset, session)
        
    # # view dataset props
    # view = dataset.view()
    # print(view)

    # print()
    
    # Sort by file path
    # sorted_view = dataset.sort_by("filepath")
    # session.view = sorted_view
    # print()
    


def data_selection(dataset, session):

    while True:
        print("Enter 'n' to quit or choose a data selcetion option")
        selection_input = input('Create a list by selection [1] or by tag [2] ?  ')

        if selection_input == 'n':
            break

        # Selected images view
        elif selection_input == '1':
            selected_ids = session.selected
            selected_view = dataset.select(selected_ids)

        # Tagged images view
        elif selection_input == '2':
            tag_name = input('Enter tag name: ')
            selected_view = dataset.match_tags(tag_name)

        else:
            print('Try again')
            continue

        print(f'Selected {len(selected_view)} images')
        if len(selected_view) > 0:

            list_file_name = input('enter the txt file name for saving the list  ')
            if list_file_name[-4:] != '.txt':
                    list_file_name = list_file_name + '.txt'

            # Write view samples paths to txt file
            # text_file_path = f'output/{session_name}/{data_type}_{cam_type_file}_list.txt'
            export_view_to_txt(selected_view, list_file_name)
     
def main(folder_dir, cfg):

    
    # config
    classes_dict = cfg['classes_dict']
    fiftyone_classes_dict = cfg['fiftyone_classes_dict']
    
    full_img_files = []
    full_label_files = []
    folders_to_ignore = ['backup', '__ToClassify', 'Verified_crops_cl_09.07.19', 'Verified_crops_cl_27.06.19']
    
    if (os.path.isfile(folder_dir)) and (".txt" in folder_dir):
        with open(folder_dir) as f:
            full_img_files = f.read().splitlines()
    else:
        full_img_files = get_files_list(folder_dir, type='image')
        full_label_files = get_files_list(folder_dir, type='label')

    print(f"total img_files: {len(full_img_files)}")
    print(f"total label_files: {len(full_label_files)}")
    
    if len(full_img_files) == 0:
        print(f"No data found in: {folder_dir}")
        exit()
    elif len(full_label_files) == 0:
        print(f"\nNo labels data found")
        user_input = input('continue without labels? (y/n)')
        if user_input.lower() == 'y':
            cfg['with_labels'] = False

    # verifiy list and labels
    if cfg['with_labels']:
        verified_list = check_list(full_img_files)
    else:
        verified_list = full_img_files

    # verified_list = verified_list[:1000]
    print("\n\n=================================== Info ==============================\n")
    print(f"verified img with labels files: {len(verified_list)} from total: {len(full_img_files)} files")
    print(f'\nLoading Fiftyone with {len(verified_list)} files')

    # sort by file name
    verified_list = sort_file_paths(verified_list)

    run_fiftyone(verified_list, cfg=cfg)
    
    return


if __name__ == '__main__':
    
    # Load yaml config file 
    with open("configs/config_carmel.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # No classes filters - to present labels as is
    cfg['classes_dict'] = None             
    cfg['fiftyone_classes_dict'] = None

    print(cfg)
    folder_dir = '/Data/Soi/Soi_field_recordings/Day/test/Records_Argaman_11.9.25'
    # folder_dir = "bad.txt"

    cfg['session_name'] = folder_dir.split('/')[-1]
    print('Starting new Session: ', cfg['session_name'])
    
    # img_files = "/home/nvidia/darknet/data/POP700_SA/day/train_File_list.txt"
        
    main(folder_dir, cfg)