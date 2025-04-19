import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import copy

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, _ = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, _ = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat , _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    # cmc, mAP, _, _, _, _, _ = evaluator.compute()
    # DJ -----------------------
    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute() 
    # cmc: (50) top 50 retreivaled images (max_rank @ function R1_mAP_eval)
    # mAP: (1)
    # distmat: [num_query, num_gallery]
    # pids: all images' label (individual id)
    # camids: all images' camera label (camera id)
    # qf: query features [num_query, 1024] (1024D)
    # gf: gallery features [num_gallery, 1024] (1024D)
    np.save("./vis/distmat", distmat)
    # file = open('./img_path_list.txt', 'w')
    # file.writelines(img_path_list)
    # file.close()
    with open("./vis/path_list.txt", 'w') as file:
        data_to_write = '\n'.join(img_path_list)    
        file.write(data_to_write)
    # DJ -----------------------

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]





import faiss
def diverse_subset_sort_kmeans(data, m, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Sorts a vector dataset into diverse subsets of size m using k-means clustering.
    Args:
        data: A PyTorch tensor or NumPy array representing the vector dataset (n x d).
        m: The desired size of each diverse subset.
    Returns:
        A PyTorch tensor or NumPy array representing the sorted dataset. Returns the same 
        type as the input data. Returns None if input is invalid.
    """
    t0 = time.time()
    if not isinstance(data, (torch.Tensor, np.ndarray)):
        print("Error: Input data must be a PyTorch tensor or NumPy array.")
        return None
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy().astype('float32')
        data_is_tensor = True
    else:
        data_np = data.astype('float32')
        data_is_tensor = False
    n, d = data_np.shape
    if m > n:
        print("Error: Subset size m cannot be larger than the total number of vectors n.")
        return None
    if m <= 0:
        print("Error: Subset size m must be a positive integer.")
        return None
    sorted_indices = []
    remaining_indices = list(range(n))
    num_subsets = (n + m - 1) // m
    # for _ in tqdm(range(num_subsets)):
    for _ in range(num_subsets):    
        subset_indices = []
        if len(remaining_indices) <= m:
            subset_indices = remaining_indices[:]
            remaining_indices = []
        else:
            remaining_data = data_np[remaining_indices]
            # FAISS KMeans - Correct and Simplified
            n_clusters = m
            if device == "cuda":
                res = faiss.StandardGpuResources()
                kmeans = faiss.Kmeans(d, n_clusters, min_points_per_centroid=1, verbose=False)
            else:
                kmeans = faiss.Kmeans(d, n_clusters, min_points_per_centroid=1, verbose=False)
            kmeans.train(remaining_data)
            _, I = kmeans.index.search(remaining_data, 1)  # Search for 1 nearest centroid for each point
            closest_indices_within_remaining = []
            for i in range(n_clusters):
                cluster_indices = np.where(I == i)[0]
                if len(cluster_indices) > 0:
                    closest_indices_within_remaining.append(cluster_indices[np.argmin(kmeans.index.search(remaining_data[cluster_indices],1)[0])])
            closest_indices_within_remaining = np.array(closest_indices_within_remaining).reshape(-1,1)
            for idx in closest_indices_within_remaining:
                subset_indices.append(remaining_indices[idx[0]])
            remaining_indices = [i for i in remaining_indices if i not in subset_indices]
        sorted_indices.extend(subset_indices)
    t1 = time.time()
    print('Spends {:.2f} seconds for diverse sorting.'.format(t1 - t0))
    return sorted_indices
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import euclidean
import random
def grouping_identical_consecutive_elements(lst):
    res = [] # initialize empty list
    temp = [lst[0]] # initialize `temp` with the first element 
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            temp.append(lst[i])
        else:
            res.append(temp)
            temp = [lst[i]]
    res.append(temp)  # Append the last group
    return res
def grouping_identical_consecutive_elements_as_ref(ref_lst, tar_lst):
    res = [] # initialize empty list
    temp = [tar_lst[0]] # initialize `temp` with the first element 
    for i in range(1, len(ref_lst)):
        if ref_lst[i] == ref_lst[i - 1]:
            temp.append(tar_lst[i])
        else:
            res.append(temp)
            temp = [tar_lst[i]]
    res.append(temp)  # Append the last group
    return res
def get_idx_in_nested_list(nested_list, target):
    for sub_list in nested_list:
        if target in sub_list:
            return (nested_list.index(sub_list), sub_list.index(target))
    # raise ValueError("'{target}' is not in list".format(char = target))
    return (-1, -1)
from PIL import Image, ImageDraw, ImageSequence, ImageFont
import io
def write_text_to_gif(gif_path, gif_text):
    input_gif = Image.open(gif_path)
    image_list = []
    # Loop over each frame in the animated image
    for frame in ImageSequence.Iterator(input_gif):
        # Draw the text on the frame  
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("../ARIAL.TTF", 40) # need to download ttf
        draw.text((10, 10), gif_text, font=font)
        del draw
        # However, 'frame' is still the animated image with many image_list. It has simply been seeked to a later frame
        # For our list of image_list, we only want the current frame
        # Saving the image without 'save_all' will turn it into a single frame image, and we can then re-open it
        # To be efficient, we will save it to a stream, rather than to file
        b = io.BytesIO()
        frame.save(b, format="GIF")
        frame = Image.open(b)
        # Then append the single frame image to a list of image_list
        image_list.append(frame)
    # Save the image_list as a new image
    image_list[0].save(gif_path, save_all=True, append_images=image_list[1:], optimize = False, duration = 500, loop = 0)
from PIL import Image
def save_as_gif(sampled_trk_path, sampled_trk_pids):
    sorted_sampled_trk_path = copy.deepcopy(sampled_trk_path)
    sorted_sampled_trk_path.sort()
    pid = sampled_trk_pids[0]
    panel_w, panel_h = 400, 400
    image_list = []
    for p in sorted_sampled_trk_path:
        img = Image.open(p)
        size_before = img.size
        img.thumbnail((panel_w,panel_h))
        panel = Image.new("RGB", (panel_w,panel_h), (20,20,20))
        panel.paste(img, (0,0))
        size_after = img.size
        # print(pid, size_before, size_after)
        image_list.append(panel)
    image_list[0].save("./vis/gallery_pid_" + str(pid).zfill(3) + ".gif", save_all=True, append_images=image_list[1:], optimize = False, duration = 500, loop = 0)
    write_text_to_gif("./vis/gallery_pid_" + str(pid).zfill(3) + ".gif", str(pid).zfill(3))
import imageio
from numpy import asarray
def save_as_video(sampled_trk_path, sampled_trk_pids):
    sorted_sampled_trk_path = copy.deepcopy(sampled_trk_path)
    sorted_sampled_trk_path.sort()
    pid = sampled_trk_pids[0]
    panel_w, panel_h = 400, 400
    image_list = []
    for p in sorted_sampled_trk_path:
        img = Image.open(p)
        size_before = img.size
        img.thumbnail((panel_w,panel_h))
        panel = Image.new("RGB", (panel_w,panel_h), (20,20,20))
        panel.paste(img, (0,0))
        size_after = img.size
        # print(pid, size_before, size_after)
        image_list.append(asarray(panel))
    writer = imageio.get_writer("./vis/gallery_pid_" + str(pid).zfill(3) + ".mp4",fps=30,codec="libx264",quality=3)   
    for frame in image_list:
        # print(frame.shape)
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()    
def do_streamed_inference_clothes(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    # here define the query set (before num_query) and gallery set (after num_query)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()
    
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    # loading images and extract the image features
    model.eval()
    img_pid_list   = []
    img_feat_list  = []
    img_camid_list = []
    img_path_list  = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat , _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_pid_list.extend(pid)
            # feat = torch.cat(feat, dim=0)                            ####################
            feat = torch.nn.functional.normalize(feat, dim=1, p=2)  # along channel ##########################################
            img_feat_list.extend(feat)
            img_camid_list.extend(camid)
            img_path_list.extend(imgpath)
        # if n_iter > 20:
        #     break    #############################
    print("img_pid_list  : ", len(img_pid_list))
    print("img_feat_list : ", len(img_feat_list))
    print("img_camid_list: ", len(img_camid_list))
    print("img_path_list : ", len(img_path_list))

 
    # packaged as camid-level groups (tracklet) per pid
    # trk_camid_list = grouping_identical_consecutive_elements(img_camid_list)                       # v1
    # trk_pid_list   = grouping_identical_consecutive_elements_as_ref(img_camid_list, img_pid_list)  # v1
    # trk_feat_list  = grouping_identical_consecutive_elements_as_ref(img_camid_list, img_feat_list) # v1
    # trk_path_list  = grouping_identical_consecutive_elements_as_ref(img_camid_list, img_path_list) # v1
    # print("Num of tracklet: ", len(trk_camid_list))
    trk_pid_list   = grouping_identical_consecutive_elements(img_pid_list)                         # CUSTOM_VIDEO
    trk_camid_list = grouping_identical_consecutive_elements_as_ref(img_pid_list, img_camid_list)  # CUSTOM_VIDEO
    trk_feat_list  = grouping_identical_consecutive_elements_as_ref(img_pid_list, img_feat_list)   # CUSTOM_VIDEO
    trk_path_list  = grouping_identical_consecutive_elements_as_ref(img_pid_list, img_path_list)   # CUSTOM_VIDEO
    print("Num of tracklets: ", len(trk_pid_list))


    # store data for visualization
    pid_list = [element for innerList in trk_pid_list for element in innerList]
    np.save("./vis/pid_array", np.array(pid_list))
    cid_list = [element for innerList in trk_camid_list for element in innerList]
    np.save("./vis/cid_array", np.array(cid_list))
    feat_list = [element for innerList in trk_feat_list for element in innerList]
    np.save("./vis/feat_array", torch.stack(feat_list, dim=0).cpu())
    path_list = [element for innerList in trk_path_list for element in innerList]
    with open("./vis/path_list.txt", "w") as file:
        data_to_write = "\n".join(path_list)    
        file.write(data_to_write)
 
    # shuffle the feeding order
    idx_list = [i for i in range(len(trk_feat_list))]
    # random.shuffle(idx_list) #################
    trk_pid_list   = [trk_pid_list[i]   for i in idx_list]
    trk_camid_list = [trk_camid_list[i] for i in idx_list]
    trk_feat_list  = [trk_feat_list[i]  for i in idx_list]
    trk_path_list  = [trk_path_list[i]  for i in idx_list]
    # print(trk_pid_list)
    



    # remove old gifs and videos
    my_dir = "./vis" # enter the dir name
    for f in os.listdir(my_dir):
        if f.startswith("gallery_pid_"):
            os.remove(os.path.join(my_dir, f))


    # dynamic gallery version 1 ----------------------------------------------------------------------------------------
    num_gallery = 100
    num_gallery_samples = 5 # 20 for 3-videos
    # define the first gallery element ---------------
    first_trk = 0
    size_trk = len(trk_feat_list[first_trk])
    # case 1: random
    # sortedIndex = [i for i in range(len(trk_feat_list[first_trk]))]
    # random.shuffle(sortedIndex)
    # case 2: diverse sort
    sortedIndex = diverse_subset_sort_kmeans(torch.stack(trk_feat_list[first_trk]), min(num_gallery_samples,size_trk))
    trk_feat_list[first_trk] = [trk_feat_list[first_trk][i] for i in sortedIndex]
    trk_pid_list[first_trk]  = [trk_pid_list[first_trk][i] for i in sortedIndex]
    trk_path_list[first_trk] = [trk_path_list[first_trk][i] for i in sortedIndex]
    sampled_trk_feat = trk_feat_list[first_trk][:min(num_gallery_samples,size_trk)]
    sampled_trk_pids = trk_pid_list[first_trk][:min(num_gallery_samples,size_trk)]
    sampled_trk_path = trk_path_list[first_trk][:min(num_gallery_samples,size_trk)]
    save_as_gif(sampled_trk_path, sampled_trk_pids)
    save_as_video(trk_path_list[first_trk], trk_pid_list[first_trk])
    print("Tracklet's PID: ", sampled_trk_pids[0], "!"+"--"*20)

    gallery_feat_list = [] # store 10 samples in feature formate per trk
    gallery_pid_list  = [] # store grouped pids
    gallery_path_list = [] # store grouped image paths
    gallery_feat_list.append(sampled_trk_feat)
    gallery_pid_list.append(sampled_trk_pids)
    gallery_path_list.append(sampled_trk_path)
    false_positive = 0
    for trk in range(1, len(trk_feat_list)):
        print("Tracklet ", trk, "!"+"--"*20)

        gallery_feats = [element for innerList in gallery_feat_list for element in innerList] # expansion
        gallery_pids  = [element for innerList in gallery_pid_list  for element in innerList] # expansion
        gallery_paths = [element for innerList in gallery_path_list for element in innerList] # expansion
        if True: # sampling some features/frames to represent the pid(trk)
            size_trk = len(trk_feat_list[trk])
            # case 1: random
            # sortedIndex = [i for i in range(len(trk_feat_list[trk]))]
            # random.shuffle(sortedIndex)
            # case 2: diverse sort
            sortedIndex = diverse_subset_sort_kmeans(torch.stack(trk_feat_list[trk]), min(num_gallery_samples,size_trk))
            trk_feat_list[trk] = [trk_feat_list[trk][i] for i in sortedIndex]
            trk_pid_list[trk]  = [trk_pid_list[trk][i]  for i in sortedIndex]
            trk_path_list[trk] = [trk_path_list[trk][i] for i in sortedIndex]
            # random.shuffle(trk_feat_list[trk])
            sampled_trk_feat = trk_feat_list[trk][:min(num_gallery_samples,size_trk)]
            sampled_trk_pids = trk_pid_list[trk][:min(num_gallery_samples,size_trk)]
            sampled_trk_path = trk_path_list[trk][:min(num_gallery_samples,size_trk)]
            save_as_gif(sampled_trk_path, sampled_trk_pids)
            save_as_video(trk_path_list[trk], trk_pid_list[trk])

        query_feats = sampled_trk_feat
        query_pids  = sampled_trk_pids
        query_paths = sampled_trk_path
        num_query = len(query_feats)
        all_feats = torch.stack(gallery_feats + query_feats)
        all_pids  = gallery_pids + query_pids
        aii_paths = gallery_paths + query_paths

        # do clustering to select the gallery samples
        # clustering_model = KMeans(n_clusters=len(gallery_feat_list)+1, n_init='auto')
        # clustering_model = AgglomerativeClustering(n_clusters=len(gallery_feat_list)+1,metric="euclidean",linkage="ward")
        clustering_model = AgglomerativeClustering(n_clusters=len(gallery_feat_list)+1,metric="euclidean",linkage="single")
        # clustering_model = AgglomerativeClustering(n_clusters=len(gallery_feat_list)+1,metric="euclidean",linkage="complete")
        # clustering_model = AgglomerativeClustering(n_clusters=len(gallery_feat_list)+1,metric="euclidean",linkage="average")
        clustering_model.fit(all_feats.cpu())
        # print(clustering_model.labels_) # the cluster id per image
        print("Tracklet's PID: ", query_pids[0], "!"+"--"*20)

        gq_cls, gq_counts = np.unique(clustering_model.labels_, return_counts=True)
        cls_gallery       = clustering_model.labels_[0:-num_query]
        g_cls, g_counts   = np.unique(cls_gallery, return_counts=True)
        cls_query         = clustering_model.labels_[-num_query:]
        q_cls, q_counts   = np.unique(cls_query, return_counts=True)
        print(" G/Q Clustering Result: \n", gq_cls, gq_counts)
        print(" G/- Clustering Result: \n", g_cls, g_counts)
        print(" -/Q Clustering Result: \n", q_cls, q_counts)
        # breakpoint()

        no_new_cluster = len(gq_cls) == len(g_cls)
        pid_this_trk   = query_pids[0]
        # query merge into gallery ------------------------------------------------------
        if no_new_cluster:
            # update gallery
            # print("  DO MERGE ---")

            target_cluster = q_cls[np.argmax(q_counts)]                                      # the cluster for merging into
            indices_target_cluster = np.where(clustering_model.labels_ == target_cluster)[0] # get all indices of points assigned to this cluster:
            target_cluster_pids = [ all_pids[i] for i in indices_target_cluster ]            # all pids in the target cluster
            
            if pid_this_trk not in target_cluster_pids:
                false_positive += 1 

            pids, pid_counts = np.unique(target_cluster_pids, return_counts=True)         # target cluster may contain noisy pids, so take the pid of the largest group
            target_pid = pids[np.argmax(pid_counts)]                                      # the pid for merging into
  
            idx_sublist, idx_list = get_idx_in_nested_list(gallery_pid_list, target_pid) 
            print(idx_sublist, idx_list, ">>>>", gallery_pid_list, target_pid)

            if idx_sublist == -1:
                print("  DO MERGE Yet APPEND ---")
                # breakpoint()
                # the current gallery not contain the target_pid, so append
                gallery_feat_list.append(query_feats)
                gallery_pid_list.append(query_pids)
            else:
                print("  DO MERGE ---")
                # the current gallery indeed contain the target_pid, so merge
                gallery_feat_list[idx_sublist].extend(query_feats)
                gallery_pid_list[idx_sublist].extend(query_pids)
 
        # query as new gallery ------------------------------------------------------
        else:
            # add new gallery
            print("  DO APPEND ---")
            gallery_feat_list.append(query_feats)
            gallery_pid_list.append(query_pids)

        print(" >> Length of gallery_feat_list", len(gallery_feat_list))
        # breakpoint()

    
    print(" >> Gallery PID : ")
    gallery_groups = [ np.unique(l).tolist() for l in  gallery_pid_list]
    print(gallery_groups)
    print(" >> False positive: ", false_positive)
    num_pids_in_gallery = len(np.unique([element for innerList in gallery_pid_list for element in innerList])) # expansion
    print(" >> num_pids_in_gallery: ", num_pids_in_gallery)

    
    from PIL import ImageSequence
    import math
    def merge_crop_gifs(merged_pids,tag):
        # make sure that all GIFs have the same width and height
        num_merged_pids = len(merged_pids)
        max_column = min(8, num_merged_pids)
        gif_list = [Image.open("./vis/gallery_pid_" + str(i).zfill(3) + ".gif")  for i in  merged_pids]
        max_frames = max([gif.n_frames for gif in gif_list])   
        panel_w = gif_list[0].width
        panel_h = gif_list[0].height
        image_list = []
        for frame_idx in range(max_frames):
            panel = Image.new("RGB", (panel_w*max_column,panel_h*math.ceil(num_merged_pids/max_column)), (20,20,20))
            for i in range(num_merged_pids):
                row_idx = i // max_column
                col_idx = i %  max_column
                try:
                    this_gif_frame = ImageSequence.Iterator(gif_list[i])[frame_idx]
                except:
                    this_gif_frame = Image.new("RGB", (panel_w,panel_h), (20,20,20))
                panel.paste(this_gif_frame, (0+col_idx*panel_w,0+row_idx*panel_h))
            image_list.append(panel)
        image_list[0].save("./vis/gallery_" + tag + "_" + "_".join([str(i) for i in merged_pids]) + ".gif", save_all = True, append_images = image_list[1:], optimize = False, duration = 500, loop = 0)    
        return 0
    def merge_crop_videos(merged_pids,tag):
        # make sure that all Videos have the same width and height
        num_merged_pids = len(merged_pids)
        max_column = min(5, num_merged_pids)
        input_video_list = [cv2.VideoCapture("./vis/gallery_pid_" + str(i).zfill(3) + ".mp4")  for i in  merged_pids]
        max_frames = int(max([video.get(cv2.CAP_PROP_FRAME_COUNT) for video in input_video_list]))   
        panel_w = int(input_video_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        panel_h = int(input_video_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = input_video_list[0].get(cv2.CAP_PROP_FPS)
        # Define the codec and create VideoWriter object
        new_video_path = "./vis/gallery_" + tag + "_" + "_".join([str(i) for i in merged_pids]) + ".mp4"
        writer = imageio.get_writer(new_video_path, fps=fps, codec="libx264", quality=3)
        last_frame_list = [-1 for i in range(num_merged_pids)]
        for frame_idx in range(max_frames):
            panel = np.zeros((panel_h*math.ceil(num_merged_pids/max_column), panel_w*max_column, 3), dtype=np.uint8)
            for i in range(num_merged_pids):
                row_idx = i // max_column
                col_idx = i %  max_column
                input_video_list[i].set(1,frame_idx)
                ret, this_gif_frame = input_video_list[i].read()
                if not ret:
                    # this_gif_frame = np.zeros((panel_h,panel_w,3), dtype=np.uint8)
                    this_gif_frame = last_frame_list[i]
                else:
                    last_frame_list[i] = this_gif_frame
                panel[0+row_idx*panel_h:0+row_idx*panel_h+panel_h, 0+col_idx*panel_w:0+col_idx*panel_w+panel_w, :] = this_gif_frame
            writer.append_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        # After the loop ends, release the video capture and writer objects and close all windows
        [video.release()   for video in  input_video_list]
        cv2.destroyAllWindows() 
        writer.close()
        return 0
    def merge_inlay_videos(merged_pids,tag):
        # make sure that all Videos have the same width and height     /local4TB/projects/dingjie/SOLIDER/VIDEO-REID/output/reid_CUSTOM_VIDEO/tracklet_001_inlay_frames.mp4
        num_merged_pids = len(merged_pids)
        max_column = min(5, num_merged_pids)
        input_video_list = [cv2.VideoCapture("../VIDEO-REID/output/reid_CUSTOM_VIDEO/tracklet_" + str(i).zfill(3) + "_inlay_frames.mp4")  for i in  merged_pids]
        max_frames = int(max([video.get(cv2.CAP_PROP_FRAME_COUNT) for video in input_video_list]))   
        panel_w = int(input_video_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        panel_h = int(input_video_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = input_video_list[0].get(cv2.CAP_PROP_FPS)
        # Define the codec and create VideoWriter object
        new_video_path = "./vis/demo_gallery_" + tag + "_" + "_".join([str(i) for i in merged_pids]) + ".mp4"
        writer = imageio.get_writer(new_video_path, fps=fps, codec="libx264", quality=3)
        last_frame_list = [-1 for i in range(num_merged_pids)]
        for frame_idx in range(max_frames):
            panel = np.zeros((panel_h*math.ceil(num_merged_pids/max_column), panel_w*max_column, 3), dtype=np.uint8)
            for i in range(num_merged_pids):
                row_idx = i // max_column
                col_idx = i %  max_column
                input_video_list[i].set(1,frame_idx)
                ret, this_gif_frame = input_video_list[i].read()
                if not ret:
                    # this_gif_frame = np.zeros((panel_h,panel_w,3), dtype=np.uint8)
                    this_gif_frame = last_frame_list[i]
                else:
                    last_frame_list[i] = this_gif_frame
                panel[0+row_idx*panel_h:0+row_idx*panel_h+panel_h, 0+col_idx*panel_w:0+col_idx*panel_w+panel_w, :] = this_gif_frame
            writer.append_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        # After the loop ends, release the video capture and writer objects and close all windows
        [video.release()   for video in  input_video_list]
        cv2.destroyAllWindows() 
        writer.close()
        return 0


    # remove old gifs
    my_dir = "./vis" # enter the dir name
    for f in os.listdir(my_dir):
        if f.startswith("gallery_group_"):
            os.remove(os.path.join(my_dir, f))
        if f.startswith("gallery_single_"):
            os.remove(os.path.join(my_dir, f)) 
        # if f.startswith("demo_"):
        #     os.remove(os.path.join(my_dir, f))                        

    num_gallery_groups = len(gallery_groups)
    single_pids = []
    for i in range(num_gallery_groups):
        merged_pids = gallery_groups[i]
        if len(merged_pids) == 1:
            single_pids.extend(merged_pids)
        else:    
            merge_crop_gifs(merged_pids,"group")
            # merge_crop_videos(merged_pids,"group")
            # merge_inlay_videos(merged_pids,"group")
    single_pids.sort()
    merge_crop_gifs(single_pids,"single")
    # merge_crop_videos(single_pids,"single")
    # merge_inlay_videos(single_pids,"single") # costy
    print(" >> num_gallery_groups: ", num_gallery_groups)


    # # store data for visualization
    # pid_list = [element for innerList in trk_pid_list for element in innerList]
    # np.save("./vis/pid_array", np.array(pid_list))

    return 1, 1

    # evaluator.update ---------------
    # def update(self, output):  # called once for each batch
    #         feat, pid, camid = output
    #         self.feats.append(feat.cpu())
    #         self.pids.extend(np.asarray(pid))
    #         self.camids.extend(np.asarray(camid))


    # cmc, mAP, _, _, _, _, _ = evaluator.compute()
    # DJ -----------------------
    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute() 
    # cmc: (50) top 50 retreivaled images (max_rank @ function R1_mAP_eval)
    # mAP: (1)
    # distmat: [num_query, num_gallery]
    # pids: all images' label (individual id)
    # camids: all images' camera label (camera id)
    # qf: query features [num_query, 1024] (1024D)
    # gf: gallery features [num_gallery, 1024] (1024D)
    np.save("distmat", distmat)
    # file = open('./img_path_list.txt', 'w')
    # file.writelines(img_path_list)
    # file.close()
    with open("./img_path_list.txt", 'w') as file:
        data_to_write = '\n'.join(img_path_list)    
        file.write(data_to_write)
    # DJ -----------------------

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]












 
# from deepface import DeepFace
from deepface.modules.detection import detect_faces
from deepface.DeepFace import represent
def do_streamed_inference_faces(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    # here define the query set (before num_query) and gallery set (after num_query)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()
    
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    # loading images and extract the image features
    model.eval()
    img_pid_list   = []
    img_feat_list  = []
    img_camid_list = []
    img_path_list  = []
    img_pick_list  = []
    minimum_face_side = 40
    minimum_face_confidence = 0.9
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            # feat , _ = model(img, cam_label=camids, view_label=target_view)
            # feat = torch.nn.functional.normalize(feat, dim=1, p=2)  # along channel ##########################################

            #######################################
            batch_size = img.shape[0]
            pick_list  = []
            feat_list  = []
            for i in range(batch_size):
                input_img = cv2.imread(imgpath[i])
                face_result = detect_faces("opencv", input_img)
                if face_result!=[]:
                    # print(face_result[0])
                    # breakpoint()
                    # print(face_result[0].facial_area)
                    # print(face_result[0].facial_area.x, face_result[0].facial_area.y, face_result[0].facial_area.w, face_result[0].facial_area.h, face_result[0].confidence)
                    x = face_result[0].facial_area.x
                    y = face_result[0].facial_area.y
                    w = face_result[0].facial_area.w
                    h = face_result[0].facial_area.h
                    c = face_result[0].confidence
                    if c < minimum_face_confidence:
                        cv2.putText(input_img, "LowC", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        pick_list.append(False)
                        feat_list.append(-1)
                        cv2.imwrite(f"./face_detection_lowc.jpg", input_img)    
                        continue

                    if w > minimum_face_side and h > minimum_face_side:
                        cv2.rectangle(input_img, (x,y), (x+w,y+h), color=(0, 255, 0), thickness=int(2))
                        face_embedding = represent(face_result[0].img, model_name="Facenet", enforce_detection=False)
                        pick_list.append(True)
                        feat_list.append(torch.tensor(face_embedding[0]["embedding"]))
                        cv2.imwrite(f"./face_detection_success.jpg", input_img)
                    else:
                        cv2.putText(input_img, "Small", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        pick_list.append(False)
                        feat_list.append(-1)
                        cv2.imwrite(f"./face_detection_small.jpg", input_img)
                else:
                    cv2.putText(input_img, "None", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    pick_list.append(False)
                    feat_list.append(-1)
                    cv2.imwrite(f"./face_detection_none.jpg", input_img)    

            img_pick_list.extend(pick_list)
            #######################################
            # pid = [pid[i] for i in pick_list if i]
            # breakpoint()

            # evaluator.update((feat, pid, camid))
            img_pid_list.extend(pid)
            # img_feat_list.extend(feat)
            img_feat_list.extend(feat_list)
            img_camid_list.extend(camid)
            img_path_list.extend(imgpath)
 
    pids, pid_counts = np.unique(img_pid_list, return_counts=True)
    print("Statistic of tracklets before Face DET -------------")
    print("Track ID: ", pids)
    print("Frame #:  ", pid_counts) 

    len_pick_list = len(img_pick_list)
    img_pid_list   = [img_pid_list[i]   for i in range(len_pick_list) if img_pick_list[i]]
    img_feat_list  = [img_feat_list[i]  for i in range(len_pick_list) if img_pick_list[i]]
    img_camid_list = [img_camid_list[i] for i in range(len_pick_list) if img_pick_list[i]]
    img_path_list  = [img_path_list[i]  for i in range(len_pick_list) if img_pick_list[i]]

        # if n_iter > 20:
        #     break    #############################
    print("img_pid_list  : ", len(img_pid_list))
    print("img_feat_list : ", len(img_feat_list))
    print("img_camid_list: ", len(img_camid_list))
    print("img_path_list : ", len(img_path_list))
 
    # packaged as camid-level groups (tracklet) per pid
    # trk_camid_list = grouping_identical_consecutive_elements(img_camid_list)                       # v1
    # trk_pid_list   = grouping_identical_consecutive_elements_as_ref(img_camid_list, img_pid_list)  # v1
    # trk_feat_list  = grouping_identical_consecutive_elements_as_ref(img_camid_list, img_feat_list) # v1
    # trk_path_list  = grouping_identical_consecutive_elements_as_ref(img_camid_list, img_path_list) # v1
    # print("Num of tracklet: ", len(trk_camid_list))
    trk_pid_list   = grouping_identical_consecutive_elements(img_pid_list)                         # CUSTOM_VIDEO
    trk_camid_list = grouping_identical_consecutive_elements_as_ref(img_pid_list, img_camid_list)  # CUSTOM_VIDEO
    trk_feat_list  = grouping_identical_consecutive_elements_as_ref(img_pid_list, img_feat_list)   # CUSTOM_VIDEO
    trk_path_list  = grouping_identical_consecutive_elements_as_ref(img_pid_list, img_path_list)   # CUSTOM_VIDEO
    print("Num of tracklets: ", len(trk_pid_list))


    # store data for visualization
    pid_list = [element for innerList in trk_pid_list for element in innerList]
    np.save("./vis/pid_array", np.array(pid_list))
    cid_list = [element for innerList in trk_camid_list for element in innerList]
    np.save("./vis/cid_array", np.array(cid_list))
    feat_list = [element for innerList in trk_feat_list for element in innerList]
    np.save("./vis/feat_array", torch.stack(feat_list, dim=0).cpu())
    path_list = [element for innerList in trk_path_list for element in innerList]
    with open("./vis/path_list.txt", "w") as file:
        data_to_write = "\n".join(path_list)    
        file.write(data_to_write)


    pids, pid_counts = np.unique(pid_list, return_counts=True)
    print("Statistic of tracklets afte Face DET-------------")
    print("Track ID: ", pids)
    print("Frame #:  ", pid_counts, "\n\n")
    

 
    # shuffle the feeding order
    idx_list = [i for i in range(len(trk_feat_list))]
    # random.shuffle(idx_list) #################
    trk_pid_list   = [trk_pid_list[i]   for i in idx_list]
    trk_camid_list = [trk_camid_list[i] for i in idx_list]
    trk_feat_list  = [trk_feat_list[i]  for i in idx_list]
    trk_path_list  = [trk_path_list[i]  for i in idx_list]
    # print(trk_pid_list)
    



    # remove old gifs and videos
    my_dir = "./vis" # enter the dir name
    for f in os.listdir(my_dir):
        if f.startswith("gallery_pid_"):
            os.remove(os.path.join(my_dir, f))


    # dynamic gallery version 1 ----------------------------------------------------------------------------------------
    num_gallery = 100
    num_gallery_samples = 5 # 20 for 3-videos
    # define the first gallery element ---------------
    first_trk = 0
    size_trk = len(trk_feat_list[first_trk])
    # case 1: random
    sortedIndex = [i for i in range(len(trk_feat_list[first_trk]))]
    # random.shuffle(sortedIndex)
    # case 2: diverse sort
    # sortedIndex = diverse_subset_sort_kmeans(torch.stack(trk_feat_list[first_trk]), min(num_gallery_samples,size_trk))
    trk_feat_list[first_trk] = [trk_feat_list[first_trk][i] for i in sortedIndex]
    trk_pid_list[first_trk]  = [trk_pid_list[first_trk][i] for i in sortedIndex]
    trk_path_list[first_trk] = [trk_path_list[first_trk][i] for i in sortedIndex]
    sampled_trk_feat = trk_feat_list[first_trk][:min(num_gallery_samples,size_trk)]
    sampled_trk_pids = trk_pid_list[first_trk][:min(num_gallery_samples,size_trk)]
    sampled_trk_path = trk_path_list[first_trk][:min(num_gallery_samples,size_trk)]
    save_as_gif(sampled_trk_path, sampled_trk_pids)
    save_as_video(trk_path_list[first_trk], trk_pid_list[first_trk])
    print("Tracklet's PID: ", sampled_trk_pids[0], "!"+"--"*20)

    gallery_feat_list = [] # store 10 samples in feature formate per trk
    gallery_pid_list  = [] # store grouped pids
    gallery_path_list = [] # store grouped image paths
    gallery_feat_list.append(sampled_trk_feat)
    gallery_pid_list.append(sampled_trk_pids)
    gallery_path_list.append(sampled_trk_path)
    false_positive = 0
    for trk in range(1, len(trk_feat_list)):
        print("Tracklet ", trk, "!"+"--"*20)

        gallery_feats = [element for innerList in gallery_feat_list for element in innerList] # expansion
        gallery_pids  = [element for innerList in gallery_pid_list  for element in innerList] # expansion
        gallery_paths = [element for innerList in gallery_path_list for element in innerList] # expansion
        if True: # sampling some features/frames to represent the pid(trk)
            size_trk = len(trk_feat_list[trk])
            # case 1: random
            sortedIndex = [i for i in range(len(trk_feat_list[trk]))]
            # random.shuffle(sortedIndex)
            # case 2: diverse sort
            # sortedIndex = diverse_subset_sort_kmeans(torch.stack(trk_feat_list[trk]), min(num_gallery_samples,size_trk))
            trk_feat_list[trk] = [trk_feat_list[trk][i] for i in sortedIndex]
            trk_pid_list[trk]  = [trk_pid_list[trk][i]  for i in sortedIndex]
            trk_path_list[trk] = [trk_path_list[trk][i] for i in sortedIndex]
            # random.shuffle(trk_feat_list[trk])
            sampled_trk_feat = trk_feat_list[trk][:min(num_gallery_samples,size_trk)]
            sampled_trk_pids = trk_pid_list[trk][:min(num_gallery_samples,size_trk)]
            sampled_trk_path = trk_path_list[trk][:min(num_gallery_samples,size_trk)]
            save_as_gif(sampled_trk_path, sampled_trk_pids)
            save_as_video(trk_path_list[trk], trk_pid_list[trk])

        query_feats = sampled_trk_feat
        query_pids  = sampled_trk_pids
        query_paths = sampled_trk_path
        num_query = len(query_feats)
        all_feats = torch.stack(gallery_feats + query_feats)
        all_pids  = gallery_pids + query_pids
        aii_paths = gallery_paths + query_paths

        # do clustering to select the gallery samples
        # clustering_model = KMeans(n_clusters=len(gallery_feat_list)+1, n_init='auto')
        # clustering_model = AgglomerativeClustering(n_clusters=len(gallery_feat_list)+1,metric="euclidean",linkage="ward")
        clustering_model = AgglomerativeClustering(n_clusters=len(gallery_feat_list)+1,metric="euclidean",linkage="single")
        # clustering_model = AgglomerativeClustering(n_clusters=len(gallery_feat_list)+1,metric="euclidean",linkage="complete")
        # clustering_model = AgglomerativeClustering(n_clusters=len(gallery_feat_list)+1,metric="euclidean",linkage="average")
        clustering_model.fit(all_feats.cpu())
        # print(clustering_model.labels_) # the cluster id per image
        print("Tracklet's PID: ", query_pids[0], "!"+"--"*20)

        gq_cls, gq_counts = np.unique(clustering_model.labels_, return_counts=True)
        cls_gallery       = clustering_model.labels_[0:-num_query]
        g_cls, g_counts   = np.unique(cls_gallery, return_counts=True)
        cls_query         = clustering_model.labels_[-num_query:]
        q_cls, q_counts   = np.unique(cls_query, return_counts=True)
        print(" G/Q Clustering Result: \n", gq_cls, gq_counts)
        print(" G/- Clustering Result: \n", g_cls, g_counts)
        print(" -/Q Clustering Result: \n", q_cls, q_counts)
        # breakpoint()

        no_new_cluster = len(gq_cls) == len(g_cls)
        pid_this_trk   = query_pids[0]
        # query merge into gallery ------------------------------------------------------
        if no_new_cluster:
            # update gallery
            # print("  DO MERGE ---")

            target_cluster = q_cls[np.argmax(q_counts)]                                      # the cluster for merging into
            indices_target_cluster = np.where(clustering_model.labels_ == target_cluster)[0] # get all indices of points assigned to this cluster:
            target_cluster_pids = [ all_pids[i] for i in indices_target_cluster ]            # all pids in the target cluster
            
            if pid_this_trk not in target_cluster_pids:
                false_positive += 1 

            pids, pid_counts = np.unique(target_cluster_pids, return_counts=True)         # target cluster may contain noisy pids, so take the pid of the largest group
            target_pid = pids[np.argmax(pid_counts)]                                      # the pid for merging into
  
            idx_sublist, idx_list = get_idx_in_nested_list(gallery_pid_list, target_pid) 
            print(idx_sublist, idx_list, ">>>>", gallery_pid_list, target_pid)

            if idx_sublist == -1:
                print("  DO MERGE Yet APPEND ---")
                # breakpoint()
                # the current gallery not contain the target_pid, so append
                gallery_feat_list.append(query_feats)
                gallery_pid_list.append(query_pids)
            else:
                print("  DO MERGE ---")
                # the current gallery indeed contain the target_pid, so merge
                gallery_feat_list[idx_sublist].extend(query_feats)
                gallery_pid_list[idx_sublist].extend(query_pids)
 
        # query as new gallery ------------------------------------------------------
        else:
            # add new gallery
            print("  DO APPEND ---")
            gallery_feat_list.append(query_feats)
            gallery_pid_list.append(query_pids)

        print(" >> Length of gallery_feat_list", len(gallery_feat_list))
        # breakpoint()

    
    print(" >> Gallery PID : ")
    gallery_groups = [ np.unique(l).tolist() for l in  gallery_pid_list]
    print(gallery_groups)
    print(" >> False positive: ", false_positive)
    num_pids_in_gallery = len(np.unique([element for innerList in gallery_pid_list for element in innerList])) # expansion
    print(" >> num_pids_in_gallery: ", num_pids_in_gallery)

    
    from PIL import ImageSequence
    import math
    def merge_crop_gifs(merged_pids,tag):
        # make sure that all GIFs have the same width and height
        num_merged_pids = len(merged_pids)
        max_column = min(8, num_merged_pids)
        gif_list = [Image.open("./vis/gallery_pid_" + str(i).zfill(3) + ".gif")  for i in  merged_pids]
        max_frames = max([gif.n_frames for gif in gif_list])   
        panel_w = gif_list[0].width
        panel_h = gif_list[0].height
        image_list = []
        for frame_idx in range(max_frames):
            panel = Image.new("RGB", (panel_w*max_column,panel_h*math.ceil(num_merged_pids/max_column)), (20,20,20))
            for i in range(num_merged_pids):
                row_idx = i // max_column
                col_idx = i %  max_column
                try:
                    this_gif_frame = ImageSequence.Iterator(gif_list[i])[frame_idx]
                except:
                    this_gif_frame = Image.new("RGB", (panel_w,panel_h), (20,20,20))
                panel.paste(this_gif_frame, (0+col_idx*panel_w,0+row_idx*panel_h))
            image_list.append(panel)
        image_list[0].save("./vis/gallery_" + tag + "_" + "_".join([str(i) for i in merged_pids]) + ".gif", save_all = True, append_images = image_list[1:], optimize = False, duration = 500, loop = 0)    
        return 0
    def merge_crop_videos(merged_pids,tag):
        # make sure that all Videos have the same width and height
        num_merged_pids = len(merged_pids)
        max_column = min(5, num_merged_pids)
        input_video_list = [cv2.VideoCapture("./vis/gallery_pid_" + str(i).zfill(3) + ".mp4")  for i in  merged_pids]
        max_frames = int(max([video.get(cv2.CAP_PROP_FRAME_COUNT) for video in input_video_list]))   
        panel_w = int(input_video_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        panel_h = int(input_video_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = input_video_list[0].get(cv2.CAP_PROP_FPS)
        # Define the codec and create VideoWriter object
        new_video_path = "./vis/gallery_" + tag + "_" + "_".join([str(i) for i in merged_pids]) + ".mp4"
        writer = imageio.get_writer(new_video_path, fps=fps, codec="libx264", quality=3)
        last_frame_list = [-1 for i in range(num_merged_pids)]
        for frame_idx in range(max_frames):
            panel = np.zeros((panel_h*math.ceil(num_merged_pids/max_column), panel_w*max_column, 3), dtype=np.uint8)
            for i in range(num_merged_pids):
                row_idx = i // max_column
                col_idx = i %  max_column
                input_video_list[i].set(1,frame_idx)
                ret, this_gif_frame = input_video_list[i].read()
                if not ret:
                    # this_gif_frame = np.zeros((panel_h,panel_w,3), dtype=np.uint8)
                    this_gif_frame = last_frame_list[i]
                else:
                    last_frame_list[i] = this_gif_frame
                panel[0+row_idx*panel_h:0+row_idx*panel_h+panel_h, 0+col_idx*panel_w:0+col_idx*panel_w+panel_w, :] = this_gif_frame
            writer.append_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        # After the loop ends, release the video capture and writer objects and close all windows
        [video.release()   for video in  input_video_list]
        cv2.destroyAllWindows() 
        writer.close()
        return 0
    def merge_inlay_videos(merged_pids,tag):
        # make sure that all Videos have the same width and height     /local4TB/projects/dingjie/SOLIDER/VIDEO-REID/output/reid_CUSTOM_VIDEO/tracklet_001_inlay_frames.mp4
        num_merged_pids = len(merged_pids)
        max_column = min(5, num_merged_pids)
        input_video_list = [cv2.VideoCapture("../VIDEO-REID/output/reid_CUSTOM_VIDEO/tracklet_" + str(i).zfill(3) + "_inlay_frames.mp4")  for i in  merged_pids]
        max_frames = int(max([video.get(cv2.CAP_PROP_FRAME_COUNT) for video in input_video_list]))   
        panel_w = int(input_video_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        panel_h = int(input_video_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = input_video_list[0].get(cv2.CAP_PROP_FPS)
        # Define the codec and create VideoWriter object
        new_video_path = "./vis/demo_gallery_" + tag + "_" + "_".join([str(i) for i in merged_pids]) + ".mp4"
        writer = imageio.get_writer(new_video_path, fps=fps, codec="libx264", quality=3)
        last_frame_list = [-1 for i in range(num_merged_pids)]
        for frame_idx in range(max_frames):
            panel = np.zeros((panel_h*math.ceil(num_merged_pids/max_column), panel_w*max_column, 3), dtype=np.uint8)
            for i in range(num_merged_pids):
                row_idx = i // max_column
                col_idx = i %  max_column
                input_video_list[i].set(1,frame_idx)
                ret, this_gif_frame = input_video_list[i].read()
                if not ret:
                    # this_gif_frame = np.zeros((panel_h,panel_w,3), dtype=np.uint8)
                    this_gif_frame = last_frame_list[i]
                else:
                    last_frame_list[i] = this_gif_frame
                panel[0+row_idx*panel_h:0+row_idx*panel_h+panel_h, 0+col_idx*panel_w:0+col_idx*panel_w+panel_w, :] = this_gif_frame
            writer.append_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        # After the loop ends, release the video capture and writer objects and close all windows
        [video.release()   for video in  input_video_list]
        cv2.destroyAllWindows() 
        writer.close()
        return 0


    # remove old gifs
    my_dir = "./vis" # enter the dir name
    for f in os.listdir(my_dir):
        if f.startswith("gallery_group_"):
            os.remove(os.path.join(my_dir, f))
        if f.startswith("gallery_single_"):
            os.remove(os.path.join(my_dir, f)) 
        # if f.startswith("demo_"):
        #     os.remove(os.path.join(my_dir, f))                        

    num_gallery_groups = len(gallery_groups)
    single_pids = []
    for i in range(num_gallery_groups):
        merged_pids = gallery_groups[i]
        if len(merged_pids) == 1:
            single_pids.extend(merged_pids)
        else:    
            merge_crop_gifs(merged_pids,"group")
            # merge_crop_videos(merged_pids,"group")
            # merge_inlay_videos(merged_pids,"group")
    single_pids.sort()
    merge_crop_gifs(single_pids,"single")
    # merge_crop_videos(single_pids,"single")
    # merge_inlay_videos(single_pids,"single") # costy
    print(" >> num_gallery_groups: ", num_gallery_groups)


    # # store data for visualization
    # pid_list = [element for innerList in trk_pid_list for element in innerList]
    # np.save("./vis/pid_array", np.array(pid_list))

    return 1, 1

    # evaluator.update ---------------
    # def update(self, output):  # called once for each batch
    #         feat, pid, camid = output
    #         self.feats.append(feat.cpu())
    #         self.pids.extend(np.asarray(pid))
    #         self.camids.extend(np.asarray(camid))


    # cmc, mAP, _, _, _, _, _ = evaluator.compute()
    # DJ -----------------------
    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute() 
    # cmc: (50) top 50 retreivaled images (max_rank @ function R1_mAP_eval)
    # mAP: (1)
    # distmat: [num_query, num_gallery]
    # pids: all images' label (individual id)
    # camids: all images' camera label (camera id)
    # qf: query features [num_query, 1024] (1024D)
    # gf: gallery features [num_gallery, 1024] (1024D)
    np.save("distmat", distmat)
    # file = open('./img_path_list.txt', 'w')
    # file.writelines(img_path_list)
    # file.close()
    with open("./img_path_list.txt", 'w') as file:
        data_to_write = '\n'.join(img_path_list)    
        file.write(data_to_write)
    # DJ -----------------------

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


