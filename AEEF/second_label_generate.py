import cv2
from tqdm import tqdm
import pandas as pd

from utils.dataload import get_img_locs,calculate_changes,convert_token,binarize

img1_loc = "./Second/im1"
label_loc = "./Second/"


t1_locs = get_img_locs(img1_loc,".png")
t2_locs = [i.replace("/im1/","/im2/") for i in t1_locs]
l1_locs = [i.replace("/im1/","/label1/") for i in t1_locs]
l2_locs = [i.replace("/im1/","/label2/") for i in t1_locs]

img_nums = [int(i.split("/")[-1].split(".")[0]) for i in t1_locs]
cols = [str(i) + str(j) for i in range(1,7) for j in range(1,7)]
changes_frame = pd.DataFrame(columns=cols,index=img_nums)

pbar = tqdm(range(len(t1_locs)),total=len(t1_locs))

for i in pbar:
  img_l1 = cv2.cvtColor(cv2.imread(l1_locs[i],cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
  img_l2 = cv2.cvtColor(cv2.imread(l2_locs[i],cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

  l1_img = convert_token(img_l1)
  l2_img = convert_token(img_l2)

  changes = calculate_changes(l1_img,l2_img)
  img_num = int(l1_locs[i].split("/")[-1].split(".")[0])
  changes_frame.loc[img_num] = changes.loc['ratios']


changes_label = changes_frame.applymap(binarize)
changes_label.to_csv(label_loc+"secondclabels.csv",index_label="Img No")

print("Number of Images per Change Class")
print(changes_label.sum())

