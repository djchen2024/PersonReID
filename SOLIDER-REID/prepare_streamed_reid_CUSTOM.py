import cv2
import os
import json
from tqdm import tqdm
import numpy as np
  
 
def list_files_with_subdirs(start_path="."):
    file_list = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def list_subdirs(start_path="."):
    subdir_list = []
    for root, dirs, files in os.walk(start_path):
        for dir in dirs:
            subdir_list.append(os.path.join(root, dir))
    return subdir_list


def split_list_equally(input_list, num):
    avg = len(input_list) / float(num)
    output_list = []
    last = 0.0
    while last < len(input_list):
        output_list.append(input_list[int(last):int(last + avg)])
        last += avg
    return output_list


image_extensions = ["jpg", "JPG", "jpeg", "bmp", "png", "webp"]
def make_query_gallery(img_dir, query_dir, gallery_dir):

    num_pid = len(list_subdirs(img_dir))
    print("The number of individuals are: ", num_pid)
    
    for pid_dir in tqdm(list_subdirs(img_dir)):
        img_list = os.listdir(pid_dir)
        img_list = [fn for fn in os.listdir(pid_dir) if any(fn.endswith(ext) for ext in image_extensions)]
        pid = os.path.basename(pid_dir)

        img_list.sort()
        num_selected_img = len(img_list)

        num_split = 3
        split_list = split_list_equally(img_list, num_split)

        for id, query_list in enumerate(split_list):
            cid = "c"+ str(id+1)
            for img_path in query_list:
                # print(img_path)
                try:            
                    fid = img_path.rsplit(".", 1)[0].rsplit("_", 1)[1] 
                    img = cv2.imread(os.path.join(pid_dir, img_path))
                    img_save_name = "{}_{}_{}.jpg".format(pid.zfill(4),cid,fid.zfill(5))
                    # print(img_save_name)
                    cv2.imwrite(os.path.join(query_dir, img_save_name), img)
                except:
                    print(os.path.join(pid_dir, img_path))
                    continue
        
    return



 
# image name format: 0001_c1.jpg (0000: pid/individual, c1: camera id)
if __name__ == "__main__":

    img_dir     = "../../data/reid_testset_v1/reid_testset_v1"   
    save_dir    = "../../data/reid_CUSTOM_v1"    
    train_dir   = os.path.join(save_dir, 'bounding_box_train')
    query_dir   = os.path.join(save_dir, 'query')
    gallery_dir = os.path.join(save_dir, 'bounding_box_test')
    # clear data generated from the previous round
    import shutil
    if os.path.exists(save_dir):
        print("REMOVE old v1 ----")
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    os.makedirs(train_dir)
    os.makedirs(query_dir)
    os.makedirs(gallery_dir)
 
    make_query_gallery(img_dir, query_dir, gallery_dir) # define query and gallery sets
    print("v1 DONE ----")
 


    img_dir = "../../data/reid_testset_v2/re-id"
    save_dir = "../../data/reid_CUSTOM_v2"    
    train_dir   = os.path.join(save_dir, 'bounding_box_train')
    query_dir   = os.path.join(save_dir, 'query')
    gallery_dir = os.path.join(save_dir, 'bounding_box_test')
    
    # clear data generated from the previous round
    import shutil
    if os.path.exists(save_dir):
        print("REMOVE old v2 ----")
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    os.makedirs(train_dir)
    os.makedirs(query_dir)
    os.makedirs(gallery_dir)

    make_query_gallery(img_dir, query_dir, gallery_dir) # merge all splits
    print("v2 DONE ----")
