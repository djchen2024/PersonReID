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





image_extensions = ["jpg", "JPG", "jpeg", "bmp", "png", "webp"]
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
def make_query_gallery(img_dir, query_dir, gallery_dir):

    num_pid = len(list_subdirs(img_dir))
    print("The number of individuals are: ", num_pid)
    
    for pid_dir in tqdm(list_subdirs(img_dir)):
        img_list = os.listdir(pid_dir)
        img_list = [fn for fn in os.listdir(pid_dir) if any(fn.endswith(ext) for ext in image_extensions)]
        pid = os.path.basename(pid_dir)
        use_clustering = False


        if use_clustering:
            # extract image-level features for clustering
            num_img  = len(img_list)
            ratio = 0.1
            feature_length = 20*20*3
            img_feat = np.zeros((num_img, feature_length))
            for i, img_path in enumerate(img_list):
                try:            
                    img = cv2.imread(os.path.join(pid_dir, img_path))
                    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
                    img_feat[i, :] = img.flatten()
                except:
                    print("Unable to read: ", os.path.join(pid_dir, img_path))
                    # cv2.imwrite("resized_image.jpg", img)
                    continue  
            clustering_model = KMeans(n_clusters=max(int(num_img*ratio),1), n_init='auto')
            clustering_model.fit(img_feat)
            # print(clustering_model.labels_) # the cluster id per image
        
            # Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
            closest_pt_idx = []
            for iclust in range(len(set(clustering_model.labels_))):
                # get all points assigned to each cluster:
                cluster_pts = img_feat[clustering_model.labels_ == iclust]
                # get all indices of points assigned to this cluster:
                cluster_pts_indices = np.where(clustering_model.labels_ == iclust)[0]
                cluster_cen = clustering_model.cluster_centers_[iclust]
                min_idx = np.argmin([euclidean(img_feat[idx], cluster_cen) for idx in cluster_pts_indices])
                # print('closest point to cluster center: ', cluster_pts[min_idx])
                # print('closest index of point to cluster center: ', cluster_pts_indices[min_idx])
                # print('  ', img_feat[cluster_pts_indices[min_idx]])
                closest_pt_idx.append(cluster_pts_indices[min_idx])
            # print(closest_pt_idx)
            closest_pt_idx.sort()
            # print(closest_pt_idx)
           
            selected_img_list = []
            for idx in closest_pt_idx:
                # print(idx, img_list[idx])
                selected_img_list.append(img_list[idx])
            selected_img_list.sort()
            # print(selected_img_list)
        else:
            selected_img_list = img_list
            selected_img_list.sort()

        query_ratio = 0.3
        num_selected_img = len(selected_img_list)

        num_query    = max(int(query_ratio*num_selected_img),1)
        query_list   = selected_img_list[:num_query]
        gallery_list = selected_img_list[num_query:]


        cid = "c1"
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
        
        cid = "c2"
        for img_path in gallery_list:
            # print(img_path)
            try:
                fid = img_path.rsplit(".", 1)[0].rsplit("_", 1)[1] 
                img = cv2.imread(os.path.join(pid_dir, img_path))
                img_save_name = "{}_{}_{}.jpg".format(pid.zfill(4),cid,fid.zfill(5))
                # print(img_save_name)
                cv2.imwrite(os.path.join(gallery_dir, img_save_name), img)  
            except:
                print(os.path.join(pid_dir, img_path))
                continue

        # breakpoint()
    return



import os
import random
import shutil
from shutil import copy2
def split_data(img_dir, label_path, max_imgs=1000): # split data in a specific size
    # read GT labels
    with open(label_path, "r") as f:
        lines = f.readlines()
    num_labels = len(lines)
    print(">> Loading labels: " + str(num_labels) )
    gt_name_list = []
    gt_line_list = []
    for line in tqdm(lines):
        img_name = line.strip().split("\t")[0]
        gt_name_list.append(img_name)
        gt_line_list.append(line)

    # read images
    file_list = os.listdir(img_dir)
    num_files = len(file_list)
    print(">> Number of images: " + str(num_files) )
    index_list = list(range(num_files))
    random.shuffle(index_list)

    # Unlabeled images
    if num_labels != num_files:
        unlabeled_img_dir = img_dir.rsplit("/",1)[0]+ '/unlabeled_' + img_dir.rsplit("/",1)[1]
        os.makedirs(unlabeled_img_dir, exist_ok=True) 

    # split data into subdirs
    split_img_dir = img_dir.rsplit("/",1)[0]+ '/split_' + img_dir.rsplit("/",1)[1]
    os.makedirs(split_img_dir, exist_ok=True) 
    num_subfolder = num_files//max_imgs + 1
    for i in range(num_subfolder):
        subfolder_path = os.path.join(split_img_dir, str(i).zfill(3))
        os.makedirs(subfolder_path, exist_ok=True) 
    split_label_path = label_path.rsplit("/",1)[0] + '/split_' + label_path.rsplit("/",1)[1]
    with open(split_label_path, 'w+', encoding='utf-8') as split_label:    
        for i in tqdm(index_list):
            file_path = os.path.join(img_dir, file_list[i])
            if file_list[i] in gt_name_list:
                idx = gt_name_list.index(file_list[i])
                gt_line = gt_line_list[idx]
                # move image and change the corresponding path in label file
                copy2(file_path, os.path.join(split_img_dir, str(i//max_imgs).zfill(3)))
                # move(file_path, os.path.join(split_img_dir, str(i//max_imgs).zfill(3)))
                split_label.write(gt_line.replace(file_list[i], str(i//max_imgs).zfill(3)+"/"+file_list[i]))                
            else:
                print("No GT available: ", file_path)
                # move image and change the corresponding path in label file
                copy2(file_path, unlabeled_img_dir)
                # move(file_path, unlabeled_img_dir)
 
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
