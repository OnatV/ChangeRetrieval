import random,json,argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils.dataload import create_dataloader

from models.earlyfusion1024 import Encoder, Decoder, init_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size" ,type=int,default=2)
    parser.add_argument("--num-epoch",type=int,default=10)
    parser.add_argument("--learn-rate",type=float,default=0.001)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--train-loc", type=str, default="./train.json")
    parser.add_argument("--weights-loc", type=str, default="./weights/")
    parser.add_argument("--init-weight", type=bool, default=True)
    parser.add_argument("--checkpoint", type=int, default=1)
    parser.add_argument("--out-loc",type=str,default="./w/")
    parser.add_argument("--cache-images",type=bool,default=False)
    parser.add_argument("--ngpu", type=int, default=1)

    opt = parser.parse_args()
    print(opt)

    bsize = opt.batch_size
    num_epochs = opt.num_epoch
    lr = opt.learn_rate
    beta1 = opt.beta1
    beta2 = opt.beta2

    train_loc = opt.train_loc
    weights_loc = opt.weights_loc
    init_w = opt.init_weight
    c_numb = opt.checkpoint
    out_loc = opt.out_loc
    ngpu = opt.ngpu

    cache_imgs = opt.cache_images

    with open(train_loc) as json_file:
        t1trainimgs = json.load(json_file)

    dataloader = create_dataloader(train_loc, bsize)

    if cache_imgs:
        t1imgsdict = dict()
        t2imgsdict = dict()
        t1imgsbar = tqdm(t1trainimgs, total=len(t1trainimgs))
        for i in t1imgsbar:
            t1imgsdict[i] = cv2.cvtColor(cv2.imread(i, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            t2imgsdict[i.replace("/im1/", "/im2/")] = cv2.cvtColor(cv2.imread(i.replace("/im1/", "/im2/"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

    netE = Encoder().to(device)
    netD = Decoder().to(device)

    if init_w:
        netE.apply(init_weights)
        netD.apply(init_weights)
    else:
        netE.load_state_dict(torch.load(weights_loc + "Encoderearly1024.pt"))
        netD.load_state_dict(torch.load(weights_loc + "Decoderearly1024.pt"))

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(list(netE.parameters()) + list(netD.parameters()), lr=lr, betas=(beta1, beta2))
    imgs1 = torch.zeros((bsize, 3, 512, 512), dtype=torch.float).to(device)
    imgs2 = torch.zeros((bsize, 3, 512, 512), dtype=torch.float).to(device)
    imgs = torch.cat((imgs1, imgs2), 1)

    #tx = netE(imgs)
    #ot = netD(tx)

    print("Starting Training Loop...")
    losses = []
    netE.train()
    netD.train()

    cudnn.benchmark = True

    for epoch in range(num_epochs):

        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=len(dataloader))

        for i, img_paths in pbar:
            if len(img_paths) != bsize:
                continue
            for i, img_loc in enumerate(img_paths):
                if cache_imgs:
                    img1 = t1imgsdict[img_loc]
                    img2 = t2imgsdict[img_loc.replace("/im1/", "/im2/")]
                else:
                    img1 = cv2.cvtColor(cv2.imread(img_loc, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(cv2.imread(img_loc.replace("/im1/", "/im2/"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

                aug = random.random()
                if aug < 0.2:
                    img1 = cv2.rotate(img1, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                    img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                elif aug < 0.4:
                    img1 = cv2.rotate(img1, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif aug < 0.6:
                    img1 = cv2.rotate(img1, rotateCode=cv2.ROTATE_180)
                    img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_180)
                elif aug < 0.8:
                    img1 = cv2.flip(img1, 0)
                    img2 = cv2.flip(img2, 0)
                else:
                    img1 = cv2.flip(img1, 1)
                    img2 = cv2.flip(img2, 1)

                t1 = ((img1.astype(np.float) / 255.0) * 2) - 1
                t2 = ((img2.astype(np.float) / 255.0) * 2) - 1
                imgs1[i, :, :, :] = torch.from_numpy(t1.transpose(2, 0, 1)).to(device)
                imgs2[i, :, :, :] = torch.from_numpy(t2.transpose(2, 0, 1)).to(device)

            imgs = torch.cat((imgs1, imgs2), 1)

            netD.zero_grad()
            netE.zero_grad()

            tx = netE(imgs)
            ot = netD(tx)

            loss1 = criterion1(ot, (imgs2 - imgs1) / 2)
            loss2 = criterion2(ot, (imgs2 - imgs1) / 2)
            loss = loss1 + loss2

            losses.append("%10.4g" % loss)

            loss.backward()

            optimizer.step()

            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 2 + "%10.4g" * 2) % ("%g/%g" % (epoch, num_epochs - 1), mem, loss, imgs1.shape[-1])
            pbar.set_description(s)

        or1 = np.round((imgs1[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255)
        or2 = np.round((imgs2[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255)
        cr1 = np.round((((imgs2[0] - imgs1[0]) / 2).permute(1, 2, 0).cpu().detach().numpy() + 1) / 2 * 255)
        cr2 = np.round((ot[0].permute(1, 2, 0).cpu().detach().numpy() + 1) / 2 * 255)

        vis1 = np.concatenate((or1, or2), axis=1)
        vis2 = np.concatenate((cr1, cr2), axis=1)
        ig = np.concatenate((vis1, vis2), axis=0)
        cv2.imwrite("Epoch_" + str(epoch) + '_out.png', ig)

        if (epoch + 1) % c_numb == 0:
            torch.save(netE.state_dict(), out_loc + "Encoderearly1024.pt")
            torch.save(netD.state_dict(), out_loc + "Decoderearly1024.pt")
            with open(out_loc + "lossesearly1024.json", "w") as lossfile:
                json.dump(losses, lossfile, indent=4)

    print("Training Has Ended")
    torch.save(netE.state_dict(), out_loc + "lastEncoderearly1024.pt")
    torch.save(netD.state_dict(), out_loc + "lastDecoderearly1024.pt")
    with open(out_loc + "lossesearly1024.json", "w") as lossfile:
        json.dump(losses, lossfile, indent=4)
