'''
training on the speficed method and fold and scenarios

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
import matplotlib.pyplot as plt
from MST_sunet import mst
from models.physunet import PhySUNet
from utils.utils_dl import setup_seed, AverageMeter, NegativeMaxCrossCorr, MapPSD, concatenate_output,NegPearson,MapPearson,SNR_loss, split_clips
from utils.utils_trad import butter_bandpass, calc_hr
from utils.utils_sunet import Acc, create_datasets

f_min = 0.5
f_max = 3

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
        #yuv_mstmap = yuv_mstmap.to(device=device, dtype=torch.float)
        #rgbyuv_mstmap = torch.cat([mstmap,yuv_mstmap],dim=1)

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

        output_npy = output.detach().cpu().numpy()
        output_lin = np.zeros((output_npy.shape[0],output_npy.shape[1],int(output_npy.shape[2]/3),int(output_npy.shape[3]*3)))

        output_lin[:,:,:,0:192] = output_npy[:,:,0:64,:]
        output_lin[:,:,:,192:2*192] = output_npy[:,:,64:128,:]
        output_lin[:,:,:,2*192:] = output_npy[:,:,128:,:]


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
parser.add_argument('-l', '--loadmod', type=str, required=True)
args = parser.parse_args()
loadmod = args.loadmod.split("/")[-2].split("__")
fold = int(loadmod[2]) - 1
name = loadmod[1]
dataset = loadmod[0]

setup_seed()
#method = args.method

train_stride = 30 #doesnt matter really
seq_len = 576

alpha = 5#5
gamma = 1
delta = 5#5

BATCH_SIZE = 8
NUM_WORKERS = 2*BATCH_SIZE
if NUM_WORKERS > 10:
    NUM_WORKERS = 10

 # folds go from 1 to 5

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
model.load_state_dict(torch.load(args.loadmod))

criterion_hr = nn.L1Loss()
criterion_hr = criterion_hr.to(device)
criterion_rppg = NegativeMaxCrossCorr(180,42)
criterion_fft = MapPSD("mse")

criterion_fft = criterion_fft.to(device)
criterion_rppg = criterion_rppg.to(device)

total_epochs = 1
print('start eval...')
for epoch in range(0, total_epochs):
    losses_valid, losses_rppg_valid, losses_hr_valid, losses_fft_valid, acc_valid  = validate(valid_loader, model, epoch)
print(acc_valid.rmse)
print('finished eval')
with open("stats.txt", "a") as file_object:
    # Append 'hello' at the end of file
    file_object.write("STATS: "+args.loadmod+'  -  MAE: '+str(acc_valid.mae)+'  -  RMSE: '+str(acc_valid.rmse)+'  -  STD: '+str(acc_valid.std)+'\n')
