import os
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import cv2
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch
import math
from PIL import Image
import torch
from scipy.fft import fft,fftfreq
import torchvision.transforms.functional as transF
import heartpy as hp
from skimage.transform import resize
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
from utils.utils_trad import calc_hr

def mask(mask_size,mask_patch_size,channels,mask_ratio):
    flat_size = mask_size*mask_size
    num_neg = int(flat_size*mask_ratio)
    num_pos = flat_size-num_neg
    pos = np.ones(num_pos)
    neg = np.zeros(num_neg)
    mask = np.concatenate((pos,neg),axis=0)
    np.random.shuffle(mask)
    mask = mask.reshape((mask_size,mask_size,1))
    mask = np.repeat(mask, channels, axis=2)
    mask = resize(mask,(mask_size*mask_patch_size,mask_size*mask_patch_size,channels),order=0)
    #mask_t = torch.tensor(mask).permute(2,0,1)
    return mask

def butter_bandpass(sig, lowcut, highcut, fs, order=3):
    # butterworth bandpass filter

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y


class mst(Dataset):
    def __init__(self, data,stride,shuffle=True, Training=True, transform=None,seq_len=576):
        self.train = Training
        self.data = data
        self.transform = transform
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        dir = self.data[idx]
        shift = 0
        sr = 1
        if type(dir) is tuple:
            shift = dir[1]
            dir = dir[0]

        dataset=""
        if "VIPL" in dir:
            dataset = "vipl"
        if "OBF" in dir:
            dataset = "obf"
        if "PURE" in dir:
            dataset = "pure"
        if "MMSE" in dir:
            dataset = "mmse"

        shift = int(shift)
        mstmap = np.load(os.path.join(dir,"mstmap.npy"))[:,:,0:6]

        if dataset == "vipl":
            fps = np.load(os.path.join(dir,"fps.npy"))[0]
            bvm_map = np.load(os.path.join(dir,"bvm_map.npy"))[:,:,0:6]
            wave = bvm_map[0,:,0]
        if dataset == "obf" or dataset == "pure" or dataset == "mmse":
            wave = np.load(os.path.join(dir,"bvp.npy"))
            #ecg = np.load(os.path.join(dir,"ecg.npy"))[:len(wave)+120]
            #ecg = ecg[60:-60]
            fps = 30

        #if mstmap.shape[1]>=self.seq_len:   REMEMBER FOR VIPLHR THIS IS NOT TRUE
        mstmap = mstmap[:,int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift,:]
        wave = wave[int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift]
        #ecg = ecg[int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift]


        input_chrom = mstmap[-1,:,0:3]
        for c in range(0,3):
            temp = input_chrom[:,c]
            input_chrom[:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255

        input_chrom = CHROM_rppg(input_chrom)

        input_chrom = butter_bandpass(input_chrom, 0.7, 3, fps)
        chrom_hr,_,_,_ = calc_hr(input_chrom)

        if sr != 1:
            for idx in range(0,mstmap.shape[0]):
                for c in range(0,6):
                    temp = mstmap[idx,:,c]
                    mstmap[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;
            mstmap = mstmap.astype(np.uint8())

            mstmap = cv2.resize(mstmap, dsize=(self.seq_len,mstmap.shape[0]), interpolation=cv2.INTER_CUBIC)
            for idx in range(0,mstmap.shape[0]):
                for c in range(0,6):
                    temp = mstmap[idx,:,c]
                    mstmap[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;

        wave = (wave-np.min(wave))/(np.max(wave)-np.min(wave))
        wave = butter_bandpass(wave, 0.7, 3, fps) #low pass filter to remove DC component (introduced by normalisation)

        hr,_,_,_ = calc_hr(wave)

        if sr != 1:
            wave =  resample(wave,self.seq_len)
            wave = butter_bandpass(wave, 0.7, 3, fps)
            hr = hr / sr
        if self.seq_len == 576:

            bvpmap = np.stack([wave]*64,axis=0)
            bvpmap = np.stack([bvpmap]*6,axis=2)


            for idx in range(0,mstmap.shape[0]):
                for c in range(0,6):
                    temp = mstmap[idx,:,c]
                    mstmap[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;
            mstmap = mstmap.astype(np.uint8())

            resized_mstmap = np.zeros((64,self.seq_len,6))

            for i in range(0,self.seq_len):
                row1 = mstmap[:,[i],:]
                row = cv2.resize(row1, dsize=(1,64), interpolation=cv2.INTER_CUBIC)
                resized_mstmap[:,[i],:] = row

            for idx in range(0,resized_mstmap.shape[0]):
                for c in range(0,6):
                    temp = resized_mstmap[idx,:,c]
                    resized_mstmap[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;

        if self.seq_len == 896:

            bvpmap = np.stack([wave]*56,axis=0)
            bvpmap = np.stack([bvpmap]*6,axis=2)


            for idx in range(0,mstmap.shape[0]):
                for c in range(0,6):
                    temp = mstmap[idx,:,c]
                    mstmap[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;
            mstmap = mstmap.astype(np.uint8())

            resized_mstmap = np.zeros((56,self.seq_len,6))

            for i in range(0,self.seq_len):
                row1 = mstmap[:,[i],:]
                row = cv2.resize(row1, dsize=(1,56), interpolation=cv2.INTER_CUBIC)
                resized_mstmap[:,[i],:] = row

            for idx in range(0,resized_mstmap.shape[0]):
                for c in range(0,6):
                    temp = resized_mstmap[idx,:,c]
                    resized_mstmap[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;

        #mstmap = resized_mstmap
        if self.seq_len == 576:
            mstmap = np.zeros((192,192,6))
            mstmap[0:64,:,:] = resized_mstmap[:,0:192,:]
            mstmap[64:128,:,:] = resized_mstmap[:,192:384,:]
            mstmap[128:192,:,:] = resized_mstmap[:,384:576,:]
            stacked_bvpmap = np.zeros((192,192,6))
            stacked_bvpmap[0:64,:,:] = bvpmap[:,0:192,:]
            stacked_bvpmap[64:128,:,:] = bvpmap[:,192:384,:]
            stacked_bvpmap[128:192,:,:] = bvpmap[:,384:576,:]
        if self.seq_len == 896:
            mstmap = np.zeros((224,224,6))
            mstmap[0:56,:,:] = resized_mstmap[:,0:224,:]
            mstmap[56:112,:,:] = resized_mstmap[:,224:448,:]
            mstmap[112:168,:,:] = resized_mstmap[:,448:672,:]
            mstmap[168:224,:,:] = resized_mstmap[:,672:896,:]
            stacked_bvpmap = np.zeros((224,224,6))
            stacked_bvpmap[0:56,:,:] = bvpmap[:,0:224,:]
            stacked_bvpmap[56:112,:,:] = bvpmap[:,224:448,:]
            stacked_bvpmap[112:168,:,:] = bvpmap[:,448:672,:]
            stacked_bvpmap[168:224,:,:] = bvpmap[:,672:896,:]

        stacked_bvpmap = ((stacked_bvpmap-np.min(stacked_bvpmap))/(np.max(stacked_bvpmap)-np.min(stacked_bvpmap)))*255
        mstmap1 = mstmap[:,:,0:3].astype(np.uint8())
        #mstmap2 = mstmap[:,:,3:6].astype(np.uint8())
        bvpmap1 = stacked_bvpmap[:,:,0:3].astype(np.uint8())
        #bvpmap2 = bvpmap[:,:,3:6].astype(np.uint8())

        #masked_map = (mstmap1*mask(mask_size=48,mask_patch_size=4,channels=3,mask_ratio=0.75)).astype(np.uint8())
        masked_map = mstmap1.copy()

        mstmap1 = Image.fromarray(mstmap1)
        masked_map = Image.fromarray(masked_map)
        #mstmap2 = Image.fromarray(mstmap2)

        bvpmap1 = Image.fromarray(bvpmap1)
        #bvpmap2 = Image.fromarray(bvpmap2)
        mstmap1 = self.transform(mstmap1)
        masked_map = self.transform(masked_map)
        bvpmap1 = self.transform(bvpmap1)
        #mstmap2 = self.transform(mstmap2)
        #bvpmap2 = self.transform(bvpmap2)
        #feature_map1 = self.transform(feature_map1)
        #feature_map2 = self.transform(feature_map2)
        #mstmap = torch.cat((mstmap1, mstmap2), dim = 0);
        #bvpmap = torch.cat((bvpmap1, bvpmap2), dim = 0);

        sample = (mstmap1,masked_map,bvpmap1,hr,chrom_hr,fps,wave.copy(),idx) 
        return sample
