import os,random,json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def get_img_locs(dir, ext=".tif"):
    imgs_loc = []
    for (dirpath, _, filenames) in os.walk(dir):
        for file in filenames:
            if file.endswith(ext):
                imgs_loc.append(os.path.abspath(os.path.join(dirpath, file)).replace("\\","/"))
    return imgs_loc

def create_test_trainset(t1_locs,train_ratio,out_dir,seed):
    s_size = len(t1_locs)
    random.seed(seed)
    train_t1 = random.sample(t1_locs,int(s_size*train_ratio))
    test_t1 = [i for i in t1_locs if i not in train_t1]

    with open(out_dir+"train.json","w") as trainfile:
        json.dump(train_t1,trainfile,indent=4)
    with open(out_dir+"test.json", "w") as testfile:
        json.dump(test_t1,testfile,indent=4)

def read_augment_images(imgs,batchsize):
    t1s = np.zeros((512,512,3,batchsize))
    t2s = np.zeros((512,512,3,batchsize))
    for i,img_loc in enumerate(imgs):
        img1 = cv2.imread(img_loc)
        img2 = cv2.imread(img_loc.replace("/t1/","/t2/"))
        aug = random.random()
        if  aug < 0.2 :
            img1 = cv2.rotate(img1,rotateCode=cv2.ROTATE_90_CLOCKWISE)
            img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        elif aug < 0.4 :
            img1 = cv2.rotate(img1, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
            img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif aug < 0.6:
            img1 = cv2.rotate(img1, rotateCode=cv2.ROTATE_180)
            img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_180)
        elif aug < 0.8:
            img1 = cv2.flip(img1,0)
            img2 = cv2.flip(img2,0)
        else :
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
        t1 = ((img1.astype(np.float)/255.0)*2)-1
        t2 = ((img2.astype(np.float)/255.0)*2)-1
        t1s[:,:,:,i] = t1
        t2s[:, :,:, i] = t2
    return t1s,t2s

def create_dataloader(dir,batchsize):
    with open(dir) as json_file:
        data_locs = json.load(json_file)
    dataload = torch.utils.data.DataLoader(data_locs, shuffle=True, batch_size=batchsize)
    return dataload

def calculate_changes(l1, l2):
    cols = [str(i) + str(j) for i in range(1,7) for j in range(1,7)]
    labels = pd.DataFrame(columns=cols, index=['ratios'])
    for i in range(1,7):
        for j in range(1,7):
            ratio = np.mean(((l1 == i) & (l2 == j)).astype(np.float32))
            labels[str(i) + str(j)] = ratio
    return labels

def convert_token(limg):
  label = np.zeros((512,512),dtype=int)

  lw = (limg == [0,128,0])
  lw = lw.all(axis=2)
  nvg = (limg == [128,128,128])
  nvg = nvg.all(axis=2)
  tr = (limg == [0,255,0])
  tr = tr.all(axis=2)
  wtr = (limg == [0,0,255])
  wtr = wtr.all(axis=2)
  bld = (limg == [128,0,0])
  bld  = bld.all(axis=2)
  plg = (limg == [255,0,0])
  plg = plg.all(axis=2)

  label[lw] = 1
  label[nvg] = 2
  label[tr] = 3
  label[wtr] = 4
  label[bld] = 5
  label[plg] = 6

  return label

def get_nums(input):
  int_num = int(input.split( "/")[-1].split(".")[0])
  return int_num

def binarize(x):
  if x >1e-9:
    return 1
  else:
    return 0