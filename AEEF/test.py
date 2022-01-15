import random,json,argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics.pairwise import cosine_distances as cdist
from sklearn.metrics.pairwise import euclidean_distances as eucdist
from sklearn.metrics.pairwise import manhattan_distances as mandist

from utils.dataload import create_dataloader,get_nums

from models.earlyfusion1024 import Encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--test-loc", type=str, default="./test.json")
    parser.add_argument("--weights-loc", type=str, default="./weights/")
    parser.add_argument("--second-labels-loc", type=str, default="./Second/secondclabels.csv")

    opt = parser.parse_args()
    print(opt)

    bsize = opt.batch_size

    test_loc = opt.test_loc
    weights_loc = opt.weights_loc
    ngpu = opt.ngpu
    second_labels_loc = opt.second_labels_loc

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    p = torch.cuda.get_device_properties(device)
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

    netE = Encoder().to(device)
    netE.load_state_dict(torch.load(weights_loc + "Encoderearly1024.pt"))

    netE.eval()

    dataloader = create_dataloader(test_loc, bsize)

    latentspace = np.zeros((len(dataloader), 32 * 32))
    imgs_list = []
    imgs1 = torch.zeros((bsize, 3, 512, 512), dtype=torch.float).to(device)
    imgs2 = torch.zeros((bsize, 3, 512, 512), dtype=torch.float).to(device)

    print("Starting Showing Results Loop...")
    cudnn.benchmark = True

    latentbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, img_paths in latentbar:
            if len(img_paths) != bsize:
                continue
            t1s = np.zeros((512, 512, 3, bsize))
            t2s = np.zeros((512, 512, 3, bsize))
            img1 = cv2.cvtColor(cv2.imread(img_paths[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(img_paths[0].replace("/im1/", "/im2/"), cv2.IMREAD_UNCHANGED),
                                cv2.COLOR_BGR2RGB)
            t1 = ((img1.astype(np.float) / 255.0) * 2) - 1
            t2 = ((img2.astype(np.float) / 255.0) * 2) - 1
            t1s[:, :, :, 0] = t1
            t2s[:, :, :, 0] = t2
            imgs1 = torch.from_numpy(t1s).to(dtype=torch.float).to(device).permute(3, 2, 0, 1)
            imgs2 = torch.from_numpy(t2s).to(dtype=torch.float).to(device).permute(3, 2, 0, 1)
            imgs = torch.cat((imgs1, imgs2), 1)

            lt1 = netE(imgs)

            l1 = torch.flatten(lt1).cpu().numpy()
            latentspace[i, :] = l1
            imgs_list.append(img_paths[0])

    len_list = len(imgs_list)
    cdists = cdist(latentspace)
    cols = [str(i) for i in range(1, len_list)]
    csimretrieved_imgs = pd.DataFrame(columns=cols, index=imgs_list)
    pbar = tqdm([i for i in range(len_list)], total=len_list)
    for i in pbar:
        a = list(cdists[i, :])
        list_ind = [i for i in range(len_list)]
        zipped_l2 = zip(a, list_ind)
        sorted_zipped_l2 = sorted(zipped_l2)
        list_ind = [element for _, element in sorted_zipped_l2]
        list_ind.remove(i)
        csimretrieved_imgs.iloc[i, :] = [imgs_list[j] for j in list_ind]

    l2dists = eucdist(latentspace)
    l2simretrieved_imgs = pd.DataFrame(columns=cols, index=imgs_list)
    pbar = tqdm([i for i in range(len_list)], total=len_list)
    for i in pbar:
        a = list(l2dists[i, :])
        list_ind = [i for i in range(len_list)]
        zipped_l2 = zip(a, list_ind)
        sorted_zipped_l2 = sorted(zipped_l2)
        list_ind = [element for _, element in sorted_zipped_l2]
        list_ind.remove(i)
        l2simretrieved_imgs.iloc[i, :] = [imgs_list[j] for j in list_ind]

    l1dists = mandist(latentspace)
    l1simretrieved_imgs = pd.DataFrame(columns=cols, index=imgs_list)
    pbar = tqdm([i for i in range(len_list)], total=len_list)
    for i in pbar:
        a = list(l1dists[i, :])
        list_ind = [i for i in range(len_list)]
        zipped_l2 = zip(a, list_ind)
        sorted_zipped_l2 = sorted(zipped_l2)
        list_ind = [element for _, element in sorted_zipped_l2]
        list_ind.remove(i)
        l1simretrieved_imgs.iloc[i, :] = [imgs_list[j] for j in list_ind]

    l2simretrieved_imgs.reset_index(inplace=True)
    csimretrieved_imgs.reset_index(inplace=True)
    l1simretrieved_imgs.reset_index(inplace=True)

    csimretrieved_imgs.rename(columns={"index": "Image Name"}, inplace=True)
    csimretrieved_imgs = csimretrieved_imgs.applymap(get_nums)
    csimretrieved_imgs.set_index("Image Name", inplace=True)

    l1simretrieved_imgs.rename(columns={"index": "Image Name"}, inplace=True)
    l1simretrieved_imgs = l1simretrieved_imgs.applymap(get_nums)
    l1simretrieved_imgs.set_index("Image Name", inplace=True)

    l2simretrieved_imgs.rename(columns={"index": "Image Name"}, inplace=True)
    l2simretrieved_imgs = l2simretrieved_imgs.applymap(get_nums)
    l2simretrieved_imgs.set_index("Image Name", inplace=True)

    csimretrieved_imgs.sort_index(inplace=True)
    l1simretrieved_imgs.sort_index(inplace=True)
    l2simretrieved_imgs.sort_index(inplace=True)

    csimretrieved_imgs.head()
    l1simretrieved_imgs.head()
    l2simretrieved_imgs.head()

    img_numbs = list(csimretrieved_imgs.index)
    img_names = ["img_" + str(i) for i in img_numbs]

    labels = pd.read_csv(second_labels_loc)
    labels.set_index("Img No", inplace=True)

    shorter = [True if (i in img_numbs) else False for i in labels.index]
    labels = labels[shorter]

    labels_list = list(labels.columns)
    l2retrievedperf = {k: [] for k in labels_list}
    num_retr = 5
    for numbs in img_numbs:
        retr_img_list = l2simretrieved_imgs.loc[numbs][:num_retr]
        label_info = labels.loc[numbs]
        q_labels = [labels_list[i] for i in range(36) if label_info[i] == 1]
        pres = {k: 0 for k in q_labels}
        for i in range(num_retr):
            retr_img_num = retr_img_list[i]
            retr_label_info = labels.loc[retr_img_num]
            for label_i in q_labels:
                if retr_label_info[label_i] == 1:
                    pres[label_i] += 1 / num_retr
        for label_i in q_labels:
            l2retrievedperf[label_i].append(pres[label_i])
    for label_i in labels_list:
        if l2retrievedperf[label_i] != []:
            l2retrievedperf[label_i] = sum(l2retrievedperf[label_i]) / len(l2retrievedperf[label_i])
        else:
            l2retrievedperf[label_i] = 0

    print("L2-Retrieval Performance")
    print(l2retrievedperf)

    l1retrievedperf = {k: [] for k in labels_list}
    num_retr = 5
    for numbs in img_numbs:
        retr_img_list = l1simretrieved_imgs.loc[numbs][:num_retr]
        label_info = labels.loc[numbs]
        q_labels = [labels_list[i] for i in range(36) if label_info[i] == 1]
        pres = {k: 0 for k in q_labels}
        for i in range(num_retr):
            retr_img_num = retr_img_list[i]
            retr_label_info = labels.loc[retr_img_num]
            for label_i in q_labels:
                if retr_label_info[label_i] == 1:
                    pres[label_i] += 1 / num_retr
        for label_i in q_labels:
            l1retrievedperf[label_i].append(pres[label_i])
    for label_i in labels_list:
        if l1retrievedperf[label_i] != []:
            l1retrievedperf[label_i] = sum(l1retrievedperf[label_i]) / len(l1retrievedperf[label_i])
        else:
            l1retrievedperf[label_i] = 0

    print("L1-Retrieval Performance")
    print(l1retrievedperf)

    csimretrievedperf = {k: [] for k in labels_list}
    num_retr = 5
    for numbs in img_numbs:
        retr_img_list = csimretrieved_imgs.loc[numbs][:num_retr]
        label_info = labels.loc[numbs]
        q_labels = [labels_list[i] for i in range(36) if label_info[i] == 1]
        pres = {k: 0 for k in q_labels}
        for i in range(num_retr):
            retr_img_num = retr_img_list[i]
            retr_label_info = labels.loc[retr_img_num]
            for label_i in q_labels:
                if retr_label_info[label_i] == 1:
                    pres[label_i] += 1 / num_retr
        for label_i in q_labels:
            csimretrievedperf[label_i].append(pres[label_i])
    for label_i in labels_list:
        if csimretrievedperf[label_i] != []:
            csimretrievedperf[label_i] = sum(csimretrievedperf[label_i]) / len(csimretrievedperf[label_i])
        else:
            csimretrievedperf[label_i] = 0

    print("Cosine Similarity-Retrieval Performance")
    print(csimretrievedperf)





