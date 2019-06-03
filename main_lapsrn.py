import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from lapsrn import Net, L1_Charbonnier_loss
from dataset import DatasetFromHdf5, DatasetFromFolder
import time, math, glob
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.5, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

best_psnr = 0.

def main():

    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # opt.seed = random.randint(1, 10000)
    opt.seed = 0
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # train_set = DatasetFromHdf5("data/lap_pry_x4_small.h5")
    file_path = {}
    file_path['LR'] = '/home/tiger/Graduate/datasets/LapSRN/trainingset_pre/LR_npy'
    file_path['x2'] = '/home/tiger/Graduate/datasets/LapSRN/trainingset_pre/x2_npy'
    file_path['x4'] = '/home/tiger/Graduate/datasets/LapSRN/trainingset_pre/x4_npy'
    train_set = DatasetFromFolder(file_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = Net()
    criterion = L1_Charbonnier_loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained)) 

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    psnr_list = []

    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        psnr = train(training_data_loader, optimizer, model, criterion, epoch)
        psnr_list.append(psnr)

    whole_res = pd.DataFrame(
        data={'psnr': psnr_list},
        index=range(1, opt.nEpochs + 1)
    )
    whole_res.to_csv('results.csv', index_label='Epoch')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    global best_psnr

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}, best_psnr={:.2f}".format(epoch, optimizer.param_groups[0]["lr"], best_psnr))

    model.train()
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))

    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    avg_psnr_predicted = 0.0

    image_list = glob.glob('/home/tiger/Graduate/datasets/LapSRN/Set5' + "/*.*")
    train_bar = tqdm(training_data_loader)
    for iteration, batch in enumerate(train_bar):
        avg_psnr_bicubic = 0.0
        avg_elapsed_time = 0.0
        avg_psnr_predicted = 0.0

        input, label_x2, label_x4 = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            label_x2 = label_x2.cuda()
            label_x4 = label_x4.cuda()

        HR_2x, HR_4x = model(input)

        loss_x2 = criterion(HR_2x, label_x2)
        loss_x4 = criterion(HR_4x, label_x4)
        loss = loss_x2 + loss_x4

        optimizer.zero_grad()

        loss_x2.backward(retain_graph=True)

        loss_x4.backward()

        optimizer.step()

        for image_name in image_list:
            im_gt_y = sio.loadmat(image_name)['im_gt_y']
            im_b_y = sio.loadmat(image_name)['im_b_y']
            im_l_y = sio.loadmat(image_name)['im_l_y']

            im_gt_y = im_gt_y.astype(float)
            im_b_y = im_b_y.astype(float)
            im_l_y = im_l_y.astype(float)

            psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=4)
            avg_psnr_bicubic += psnr_bicubic

            im_input = im_l_y / 255.

            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
            if opt.cuda:
                im_input = im_input.cuda()
            start_time = time.time()
            HR_2x, HR_4x = model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            HR_4x = HR_4x.cpu()

            im_h_y = HR_4x.data[0].numpy().astype(np.float32)

            im_h_y = im_h_y * 255.
            im_h_y[im_h_y < 0] = 0
            im_h_y[im_h_y > 255.] = 255.
            im_h_y = im_h_y[0, :, :]

            psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=4)
            avg_psnr_predicted += psnr_predicted

        avg_psnr = round(avg_psnr_predicted / len(image_list), 2)
        train_bar.set_description(desc='%.2f' % (avg_psnr))
        if(avg_psnr > best_psnr):
            save_checkpoint(model, epoch-1)
            best_psnr = avg_psnr
        # if iteration%100 == 0:
        #     print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
        #     print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
    train_bar.close()
    return avg_psnr_predicted / len(image_list)


def save_checkpoint(model, epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "lapsrn_model_best_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    # print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
