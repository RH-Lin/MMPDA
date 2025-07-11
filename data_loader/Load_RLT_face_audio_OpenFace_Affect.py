from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa
from skimage import io
from skimage.transform import resize
import pickle
import torchaudio

face_frames = 32
OpenFace_frames = 64
#OpenFace_frames = 96

seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])
"""
this dataloader uses the extracted features. How to extract features please refer to Load_RLT_face_audio_wave_OpenFace_Affect.py.
Note that wave feature is not used here. If you want to use wave features, add network for it.

"""

class RLT_train(Dataset):

    def __init__(self, list_dir,  transform=None):
        with open(list_dir, 'rb') as f:
            self.data_list = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        # return format --> sample = {'video_x': video_x, 'x_affect': x_affect, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'audio_wave': audio_wave, 'videoname': videoname}

        sample = self.data_list[idx]

        # print(sample['audio_wave'].shape)
        # # Resample to the model sampling frequency. VERY IMPORTANT STEP!!!
        # sample_rate = audio_waveform.size(1)
        # w2v2_sr = 16000
        # if sample_rate != w2v2_sr:
        #     audio_waveform = torchaudio.functional.resample(audio_waveform, sample_rate, w2v2_sr)
        # sample['audio_wave'] = audio_waveform[0].unsqueeze(0)

        sample['audio_wave'] = sample['audio_wave'][:,:10000] # 489440

        if self.transform:
            sample = self.transform(sample)
        return sample

class RLT_test(Dataset):

    def __init__(self, list_dir,  transform=None):
        with open(list_dir, 'rb') as f:
            self.data_list = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        # return format --> sample = {'video_x': video_x, 'x_affect': x_affect, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'audio_wave': audio_wave, 'videoname': videoname}

        sample['audio_wave'] = sample['audio_wave'][:,:10000] # 489440
        
        sample = self.data_list[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample
