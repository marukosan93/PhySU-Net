'''
Supervised Training PhySUNet on specific dataset and fold, for SSL pre-training run pretrain_PU.py first and then load the pretrained model
'''

import argparse
import os
import pickle
import time
import more_itertools as mit
import numpy as np
import tensorboard_logger as tb_logger
from torch.utils import tensorboard
import torch
import torch.nn as nn
import torch.optim as op
import torchvision.transforms as T
from scipy.signal import welch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from scipy import signal

from MST_sunet import mst
from models.physunet import PhySUNet
from utils.utils_dl import setup_seed, AverageMeter, NegativeMaxCrossCorr, MapPSD, concatenate_output,NegPearson,MapPearson,SNR_loss, split_clips
from utils.utils_trad import butter_bandpass, calc_hr
from utils.utils_sunet import Acc, create_datasets

f_min = 0.5
f_max = 3

def train(train_loader, model, criterion_hr, criterion_rppg, criterion_fft, optimizer, epoch):

    #Run one train epoch
    fps = 30
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_hr = AverageMeter()
    losses_rppg = AverageMeter()
    losses_fft = AverageMeter()
    acc = Acc()
    #acc = Acc()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (mstmap, masked_map, bvpmap,hr_bvp,chrom_hr,fps_,bvp,idx) in enumerate(train_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        mstmap = mstmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)

        hr_bvp = hr_bvp.to(device)

        #Forward pass
        output, out_hr, feat = model(mstmap)

        loss_rppg = criterion_rppg(output,bvpmap)
        loss_fft = criterion_fft(output,bvpmap,fps,f_min,f_max)

        predict = out_hr.squeeze()
        target = hr_bvp
        target = (target-40)/140

        error = torch.abs(predict-target)*140
        acc.update(torch.Tensor(error),error.size()[0])

        loss_hr = criterion_hr(predict, target)

        loss = alpha * loss_hr + delta * loss_fft + gamma * loss_rppg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        losses.update(loss.item(), mstmap.size(0))
        losses_hr.update(loss_hr.item(), mstmap.size(0))
        losses_rppg.update(loss_rppg.item(), mstmap.size(0))
        losses_fft.update(loss_fft.item(), mstmap.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0 or i == len(train_loader) - 1:
            #print(gt_hr*140+40)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_hr {loss_hr.val:.4f} ({loss_hr.avg:.4f})\t'
                  'Loss_rppg {loss_rppg.val:.4f} ({loss_rppg.avg:.4f})\t'
                  'Loss_fft {loss_fft.val:.4f} ({loss_fft.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.mae:.4f} ({acc.rmse:.4f})\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss_hr=losses_hr, loss_rppg=losses_rppg, loss_fft=losses_fft, loss=losses,acc=acc))

    return losses.avg, losses_rppg.avg, losses_hr.avg, losses_fft.avg, acc

def validate(valid_loader, model, epoch):
    fps = 30
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_hr = AverageMeter()
    losses_rppg = AverageMeter()
    losses_fft = AverageMeter()
    acc = Acc()

    model.eval()
    end = time.time()
    for i, (mstmap, masked_map, bvpmap,hr_bvp,chrom_hr,fps_,bvp,idx) in enumerate(valid_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        mstmap = mstmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)

        hr_bvp =hr_bvp.to(device)

        #Inference
        with torch.no_grad():
            output, out_hr, feat = model(mstmap)

        loss_rppg = criterion_rppg(output,bvpmap)
        loss_fft = criterion_fft(output,bvpmap,fps,f_min,f_max)

        predict = out_hr.squeeze()
        target = hr_bvp
        target = (target-40)/140


        error = torch.abs(predict-target)*140
        acc.update(torch.Tensor(error),error.size()[0])

        loss_hr = criterion_hr(predict, target)

        loss = alpha * loss_hr + delta * loss_fft + gamma * loss_rppg

        loss = loss.float()

        losses.update(loss.item(), mstmap.size(0))
        losses_hr.update(loss_hr.item(), mstmap.size(0))
        losses_rppg.update(loss_rppg.item(), mstmap.size(0))
        losses_fft.update(loss_fft.item(), mstmap.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(valid_loader) - 1:
            # print(gt_hr*140+40)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_hr {loss_hr.val:.4f} ({loss_hr.avg:.4f})\t'
                  'Loss_rppg {loss_rppg.val:.4f} ({loss_rppg.avg:.4f})\t'
                  'Loss_fft {loss_fft.val:.4f} ({loss_fft.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.mae:.4f} ({acc.rmse:.4f})\n'.format(
                epoch, i, len(valid_loader), batch_time=batch_time,
                data_time=data_time, loss_hr=losses_hr, loss_rppg=losses_rppg, loss_fft=losses_fft, loss=losses,acc=acc))

    return losses.avg, losses_rppg.avg, losses_hr.avg, losses_fft.avg, acc



parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fold', type=str, required=True) #fold number to train on
parser.add_argument('-d','--data', type=str,required=True) #dataset to train on 
parser.add_argument('-n', '--name', type=str, required=True) #string to identify run
args = parser.parse_args()
dataset = args.data

setup_seed()
#method = args.method

train_stride = 30 
seq_len = 576

alpha = 5
gamma = 1
delta = 5

BATCH_SIZE = 8
NUM_WORKERS = 2*BATCH_SIZE
if NUM_WORKERS > 10:
    NUM_WORKERS = 10

if args.fold != "whole":
    fold = int(args.fold) - 1 # folds go from 1 to N
else:
    fold = args.fold

train_dirs, valid_dirs = create_datasets(dataset,fold,train_stride=train_stride,seq_len=seq_len,train_temp_aug=False)
transforms = [T.ToTensor()]#, T.Resize((64, seq_len))]
transforms = T.Compose(transforms)

train_dataset = mst(data=train_dirs, stride=train_stride, shuffle=True, transform=transforms, seq_len=seq_len)
valid_dataset = mst(data=valid_dirs, stride=seq_len, shuffle=False, transform=transforms, seq_len=seq_len)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

model = PhySUNet(img_size=(192,192),
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 2, 2],
                                depths_decoder=[1, 2, 2, 2],
                                num_heads=[3,6,12,24],
                                window_size=6,
                                mlp_ratio=2,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0,
                                drop_path_rate=0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
model.to(device)

criterion_hr = nn.L1Loss()
criterion_hr = criterion_hr.to(device)
criterion_rppg = NegativeMaxCrossCorr(180,42)
criterion_fft = MapPSD("mse")

criterion_fft = criterion_fft.to(device)
criterion_rppg = criterion_rppg.to(device)

optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-5, weight_decay=0.05)

total_epochs = 50
if __name__ == '__main__':
    # defined directory
    logdir = './records/logs/'+ args.data + '__' + args.name + '__' + args.fold
    model_saved_path = './records/model/'+ args.data + '__' + args.name + '__' + args.fold
    #output_saved_path = './records/output/' + method + '__' + train_dir + '__' + args.fold + '/'

    # loss_train and loss_valid must be in the same graph
    writer = {
        'loss_train': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_train')),
        'loss_rppg_train': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_rppg_train')),
        'loss_hr_train': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_hr_train')),
        'loss_fft_train': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_fft_train')),
        'loss_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_valid')),
        'loss_rppg_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_rppg_valid')),
        'loss_hr_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_hr_valid')),
        'loss_fft_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_fft_valid')),
        'RMSE_train': tensorboard.SummaryWriter(os.path.join(logdir, 'RMSE_train')),
        'RMSE_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'RMSE_valid'))
    }
    loss_best = 100

    print('start training...')
    for epoch in range(0, total_epochs):
        losses_train, losses_rppg_train, losses_hr_train, losses_fft_train, acc_train \
            = train(train_loader, model, criterion_hr, criterion_rppg, criterion_fft, optimizer, epoch)
        losses_valid, losses_rppg_valid, losses_hr_valid, losses_fft_valid, acc_valid \
            = validate(valid_loader, model, epoch)
        if losses_valid < loss_best:
            loss_best = losses_valid
            # save the model and output
            if not os.path.exists(model_saved_path):
                os.makedirs(model_saved_path)
            #if not os.path.exists(output_saved_path):
            #    os.makedirs(output_saved_path)
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(model_saved_path, 'last.pth'))

        writer['loss_train'].add_scalar("loss", losses_train, epoch)
        writer['loss_valid'].add_scalar("loss", losses_valid, epoch)
        writer['loss_rppg_train'].add_scalar("loss_rppg", losses_rppg_train, epoch)
        writer['loss_rppg_valid'].add_scalar("loss_rppg", losses_rppg_valid, epoch)
        writer['loss_hr_train'].add_scalar("loss_hr", losses_hr_train, epoch)
        writer['loss_hr_valid'].add_scalar("loss_hr", losses_hr_valid, epoch)
        writer['loss_fft_train'].add_scalar("loss_fft", losses_fft_train, epoch)
        writer['loss_fft_valid'].add_scalar("loss_fft", losses_fft_valid, epoch)
        writer['RMSE_train'].add_scalar("RMSE", acc_train.rmse, epoch)
        writer['RMSE_valid'].add_scalar("RMSE", acc_valid.rmse, epoch)

    # here to save the model and losses
    writer['loss_train'].close()
    writer['loss_valid'].close()
    writer['loss_rppg_train'].close()
    writer['loss_rppg_valid'].close()
    writer['loss_hr_train'].close()
    writer['loss_hr_valid'].close()
    writer['loss_fft_train'].close()
    writer['loss_fft_valid'].close()
    writer['RMSE_train'].close()
    writer['RMSE_valid'].close()

    print('finished training')
