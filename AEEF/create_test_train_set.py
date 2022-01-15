import argparse
from utils.dataload import get_img_locs,create_test_trainset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1img-dir",type=str,default="./Second/im1")
    parser.add_argument("--out-loc",type=str,default="./")
    parser.add_argument("--train-ratio",type=float,default=0.8)
    parser.add_argument("--train-seed",type=int,default=4)
    parser.add_argument("--img-ext",type=str,default=".png")
    opt = parser.parse_args()

    t1_locs = get_img_locs(opt.t1img_dir,opt.img_ext)
    create_test_trainset(t1_locs,opt.train_ratio,opt.out_loc,opt.train_seed)