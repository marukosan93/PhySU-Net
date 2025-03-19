import random
import torch
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch,filtfilt
import torchvision.transforms as T
import more_itertools as mit
import math
import torch.nn as nn
from MST_sunet import mst
import matplotlib.pyplot as plt
from einops import rearrange
from torch.autograd import Variable
import torch.nn.functional as F
import heartpy as hp
import pickle
from scipy import signal
from scipy.fft import fft

def hr_fft(sig, fs, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])

    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.7 / fs * sig_f.shape[0]).astype('int')
    high_idx = np.round(3 / fs * sig_f.shape[0]).astype('int')
    sig_f_original = sig_f.copy()


    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0


    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60


    """print(peak_idx1)
    print(peak_idx2)
    print(sig_f[peak_idx1])
    print(sig_f[peak_idx2])
    print("--- -")"""
    diff_peaks = np.max([sig_f[peak_idx1],sig_f[peak_idx2]])/np.min([sig_f[peak_idx1],sig_f[peak_idx2]])
    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10 and diff_peaks<2:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1
    hr = hr
    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr


def butter_bandpass(sig, lowcut, highcut, fs, order=3):
    # butterworth bandpass filter

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y

def list_dirs(dir,extension):
    r = []
    if extension == "bvm_map.npy" or extension == "bvp.npy":
        for root, dirs, files in os.walk(dir):
            for dir in dirs:
                dirpath = os.path.join(root, dir)
                for file in os.listdir(dirpath):
                    if file[-len(extension):] == extension:
                        r.append(dirpath)
                        break
    return r

def create_datasets(dataset,howmany,train_stride=576,seq_len=576,train_temp_aug=False):
    if dataset == "vipl":
        input_dir = "./MSTmaps/VIPL-HR"
        extension = "bvm_map.npy"
        list_subj = []
        for s in list(np.arange(1,108,1).astype(str)):
            s = "/p"+s+"/"
            list_subj.append(s)
    if dataset == "obf":
        input_dir = "./MSTmaps/OBF_video"
        extension = "bvp.npy"
        list_subj = []
        for s in list(np.arange(1,101,1).astype(str)):
            if len(s) == 1:
                s = "00"+s
            if len(s) == 2:
                s = "0"+s
            list_subj.append(s)
    if dataset == "pure":
        input_dir = "./MSTmaps/PURE_map"
        extension = "bvp.npy"
        list_subj = []

        for s in range(1,11):
            list_subj.append(str(s).zfill(2))
    if dataset == "mmse":
        input_dir = "./MSTmaps/MMSE_map"
        extension = "bvp.npy"
        list_subj = []

        for s in range(5,28):
            list_subj.append("F"+str(s).zfill(3))
        for s in range(1,18):
            list_subj.append("M"+str(s).zfill(3))

    all_dirnames = list_dirs(input_dir,extension)   #gets all filepaths that contain extension
    all_dirnames.sort()

    if dataset == "obf":
        all_dirnames.remove("./MSTmaps/OBF_video/073_2")  #GROUNTURHT DATA IS Wrong
        all_dirnames.remove("./MSTmaps/OBF_video/057_2")  #also weird for some reason

    blacklist_vipl = ['./MSTmaps/VIPL-HR/p10/v3/source1/video', './MSTmaps/VIPL-HR/p10/v3/source2/video', './MSTmaps/VIPL-HR/p10/v3/source3/video', './MSTmaps/VIPL-HR/p11/v4/source1/video', './MSTmaps/VIPL-HR/p11/v4/source3/video', './MSTmaps/VIPL-HR/p11/v5/source1/video', './MSTmaps/VIPL-HR/p11/v5/source3/video', './MSTmaps/VIPL-HR/p16/v8/source2/video', './MSTmaps/VIPL-HR/p22/v1/source1/video', './MSTmaps/VIPL-HR/p22/v1/source2/video', './MSTmaps/VIPL-HR/p22/v1/source3/video', './MSTmaps/VIPL-HR/p24/v6/source1/video', './MSTmaps/VIPL-HR/p24/v6/source3/video', './MSTmaps/VIPL-HR/p26/v8/source2/video', './MSTmaps/VIPL-HR/p29/v6/source2/video', './MSTmaps/VIPL-HR/p37/v8/source2/video', './MSTmaps/VIPL-HR/p37/v9/source2/video', './MSTmaps/VIPL-HR/p38/v3/source1/video', './MSTmaps/VIPL-HR/p38/v3/source2/video', './MSTmaps/VIPL-HR/p38/v3/source3/video', './MSTmaps/VIPL-HR/p40/v3/source2/video', './MSTmaps/VIPL-HR/p41/v2/source1/video', './MSTmaps/VIPL-HR/p41/v2/source3/video', './MSTmaps/VIPL-HR/p43/v3/source1/video', './MSTmaps/VIPL-HR/p43/v3/source2/video', './MSTmaps/VIPL-HR/p43/v3/source3/video', './MSTmaps/VIPL-HR/p44/v3/source1/video', './MSTmaps/VIPL-HR/p44/v3/source2/video', './MSTmaps/VIPL-HR/p44/v3/source3/video', './MSTmaps/VIPL-HR/p45/v3/source1/video', './MSTmaps/VIPL-HR/p45/v3/source2/video', './MSTmaps/VIPL-HR/p45/v3/source3/video', './MSTmaps/VIPL-HR/p46/v2/source1/video', './MSTmaps/VIPL-HR/p46/v2/source2/video', './MSTmaps/VIPL-HR/p46/v2/source3/video', './MSTmaps/VIPL-HR/p48/v9/source2/video', './MSTmaps/VIPL-HR/p49/v2/source1/video', './MSTmaps/VIPL-HR/p49/v2/source2/video', './MSTmaps/VIPL-HR/p49/v2/source3/video', './MSTmaps/VIPL-HR/p49/v3/source1/video', './MSTmaps/VIPL-HR/p49/v3/source2/video', './MSTmaps/VIPL-HR/p49/v3/source3/video', './MSTmaps/VIPL-HR/p50/v6/source1/video', './MSTmaps/VIPL-HR/p50/v6/source3/video', './MSTmaps/VIPL-HR/p51/v7/source2/video', './MSTmaps/VIPL-HR/p59/v3/source1/video', './MSTmaps/VIPL-HR/p59/v3/source2/video', './MSTmaps/VIPL-HR/p59/v3/source3/video', './MSTmaps/VIPL-HR/p68/v2/source1/video', './MSTmaps/VIPL-HR/p68/v2/source2/video', './MSTmaps/VIPL-HR/p68/v2/source3/video', './MSTmaps/VIPL-HR/p71/v7/source2/video', './MSTmaps/VIPL-HR/p83/v6/source1/video', './MSTmaps/VIPL-HR/p83/v6/source2/video', './MSTmaps/VIPL-HR/p83/v6/source3/video', './MSTmaps/VIPL-HR/p88/v2/source1/video', './MSTmaps/VIPL-HR/p88/v2/source2/video', './MSTmaps/VIPL-HR/p88/v2/source3/video', './MSTmaps/VIPL-HR/p88/v3/source1/video', './MSTmaps/VIPL-HR/p88/v3/source2/video', './MSTmaps/VIPL-HR/p88/v3/source3/video', './MSTmaps/VIPL-HR/p88/v9/source2/video', './MSTmaps/VIPL-HR/p90/v5/source2/video', './MSTmaps/VIPL-HR/p97/v2/source2/video', './MSTmaps/VIPL-HR/p97/v2/source3/video', './MSTmaps/VIPL-HR/p97/v7/source2/video', './MSTmaps/VIPL-HR/p97/v7/source3/video']
    blacklist_vipl = []
    if dataset == "vipl":
        for b in blacklist_vipl:
            all_dirnames.remove(b)

    #randomly shuffle,but with seed so that it's reproducible
    if dataset != "pure":
        np.random.seed(4)
        np.random.shuffle(list_subj)

    if howmany == 'whole':
        train_subj = list_subj
        valid_subj = []

        train_dirs = []
        valid_dirs = []


        for dir in all_dirnames:
            for s in train_subj:
                if s in dir:
                    train = True
            for s in valid_subj:
                if s in dir:
                    train = False
            if train:
                train_dirs.append(dir)
            else:
                valid_dirs.append(dir)


        train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
        #train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

        valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
        #valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

        #file = open("./folds/"+dataset+"_whole_aug2.pkl",'wb')
        file = open("./folds"+str(seq_len)+"/"+dataset+"_whole.pkl",'wb')
        pickle.dump(train_dirs, file)
        pickle.dump(valid_dirs, file)

        print(len(train_dirs))
        print(len(valid_dirs))

    if howmany == 'purezd':
        train_subj = ['01', '02', '03', '04', '05', '07']
        valid_subj = ['06', '08', '09', '10']

        train_dirs = []
        valid_dirs = []

        for dir in all_dirnames:
            for s in train_subj:
                if s in dir[:-3]:
                    train = True
            for s in valid_subj:
                if s in dir[:-3]:
                    train = False
            if train:
                train_dirs.append(dir)
            else:
                valid_dirs.append(dir)

        train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
        #train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

        valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
        #valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)


        #file = open("./folds/"+dataset+"_whole_aug2.pkl",'wb')
        file = open("./folds"+str(seq_len)+"/"+dataset+"_fold1.pkl",'wb')
        pickle.dump(train_dirs, file)
        pickle.dump(valid_dirs, file)
        print(len(train_dirs))
        print(len(valid_dirs))

    if howmany == '5fold':
        divided = ([list(x) for x in mit.divide(5,list_subj)])
        for fold in range(0,5):
            print("FOLD",fold+1)
            train_div = list(np.arange(0,5))
            train_div.remove(fold)

            train_subj = [*divided[train_div[0]], *divided[train_div[1]], *divided[train_div[2]],*divided[train_div[3]]] #train
            valid_subj = divided[fold] #validate

            train_dirs = []
            valid_dirs = []


            for dir in all_dirnames:
                for s in train_subj:
                    if s in dir:
                        train = True
                for s in valid_subj:
                    if s in dir:
                        train = False
                if train:
                    train_dirs.append(dir)
                else:
                    valid_dirs.append(dir)

            train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
            #train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

            valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
            #valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

            #file = open("./folds/"+dataset+"_fold"+str(fold+1)+"_aug2.pkl",'wb')
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(fold+1)+".pkl",'wb')
            pickle.dump(train_dirs, file)
            pickle.dump(valid_dirs, file)

            print(len(train_dirs))
            print(len(valid_dirs))

    if howmany == '3fold':
        divided = ([list(x) for x in mit.divide(3,list_subj)])
        for fold in range(0,3):
            print("FOLD",fold+1)
            train_div = list(np.arange(0,3))
            train_div.remove(fold)

            train_subj = [*divided[train_div[0]], *divided[train_div[1]]] #train
            valid_subj = divided[fold] #validate

            train_dirs = []
            valid_dirs = []

            for dir in all_dirnames:
                for s in train_subj:
                    if s in dir:
                        train = True
                for s in valid_subj:
                    if s in dir:
                        train = False
                if train:
                    train_dirs.append(dir)
                else:
                    valid_dirs.append(dir)

            train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
            #train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

            valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
            #valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

            #file = open("./folds/"+dataset+"_fold"+str(fold+1)+"_aug2.pkl",'wb')
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(fold+1)+".pkl",'wb')
            pickle.dump(train_dirs, file)
            pickle.dump(valid_dirs, file)

            print(len(train_dirs))
            print(len(valid_dirs))

    if howmany == '10fold':
        divided = ([list(x) for x in mit.divide(10,list_subj)])
        for fold in range(0,10):
            print("FOLD",fold+1)
            train_div = list(np.arange(0,10))
            train_div.remove(fold)

            train_subj = [*divided[train_div[0]], *divided[train_div[1]], *divided[train_div[2]],*divided[train_div[3]],*divided[train_div[4]],*divided[train_div[5]],*divided[train_div[6]],*divided[train_div[7]],*divided[train_div[8]]] #train
            valid_subj = divided[fold] #validate

            train_dirs = []
            valid_dirs = []


            for dir in all_dirnames:
                for s in train_subj:
                    if s in dir:
                        train = True
                for s in valid_subj:
                    if s in dir:
                        train = False
                if train:
                    train_dirs.append(dir)
                else:
                    valid_dirs.append(dir)

            train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
            #train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

            valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
            #valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

            #file = open("./folds/"+dataset+"_fold"+str(fold+1)+"_aug2.pkl",'wb')
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(fold+1)+".pkl",'wb')
            pickle.dump(train_dirs, file)
            pickle.dump(valid_dirs, file)

            print(len(train_dirs))
            print(len(valid_dirs))

    #NORMALIZE_MEAN = (0.5, 0.5, 0.5)
    #NORMALIZE_STD = (0.5, 0.5, 0.5)
    transforms = [   #add  data Augmentation
                  #T.Resize((192,192)),
                  T.ToTensor()#,
                  #T.Normalize((0.5823, 0.4994, 0.5634), (0.1492, 0.1859, 0.1953))
                  ]
    transforms = T.Compose(transforms)

    train_dataset = mst(data=train_dirs,stride=train_stride,shuffle=True, Training = True, transform=transforms,seq_len=seq_len)
    valid_dataset = mst(data=valid_dirs,stride=seq_len,shuffle=False, Training = False, transform=transforms)
    return train_dataset, valid_dataset

def window_dirs(dirs,stride,seq_len,train):
    dataset=""
    windowed_dirs = []
    for dir in dirs:
        if "VIPL" in dir:
            dataset = "vipl"
        if "OBF" in dir:
            dataset = "obf"
        mstmap = np.load(os.path.join(dir,"mstmap.npy"))[:,:,:]
        if dataset == "vipl":
            bvm_map = np.load(os.path.join(dir,"bvm_map.npy"))[:,:,0:6]
            wave = bvm_map[0,:,0]
        if dataset == "obf":
            wave = np.load(os.path.join(dir,"bvp.npy"))
        N = int(mstmap.shape[1])
        if N > seq_len:
            W = int((N-seq_len)/stride)
            for w in range(0,W+1):
                windowed_dirs.append((dir,w*stride))
        #else:                   SHOULD ADD SOMETHING INSTEAD OF THROWING AWAY SHORT CLIPS
    return windowed_dirs
