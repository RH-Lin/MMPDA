from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torchaudio
import pdb
import math
import os
import imgaug.augmenters as iaa
import pickle

face_frames = 32
OpenFace_frames = 64
# OpenFace_frames = 96

seq = iaa.Sequential([
    iaa.Add(value=(-40, 40), per_channel=True),  # Add color
    iaa.GammaContrast(gamma=(0.5, 1.5))  # GammaContrast with a gamma of 0.5 to 1.5
])

"""
this dataloader uses the extracted features. How to extract features please refer to Load_All_face_audio_wave_OpenFace_Affect.py.
Note that wave feature is not used here. If you want to use wave features, add network for it.
"""

class All_train(Dataset):

    def __init__(self, list_dir, num_av_tokens=64, transform=None):
        self.data_list = []
        for dir in list_dir:
            with open(dir, 'rb') as f:
                data_dict=pickle.load(f)
                self.data_list.extend(data_dict)
        self.transform = transform

        frame_length = []
        for i in range(len(self.data_list)):
            if self.data_list[i]['audio_x'].shape[-1] > 3:
                self.data_list[i]['audio_x']=self.data_list[i]['audio_x'][:, :, :3] # for MDPE
            frame_length.append(self.data_list[i]['audio_wave'].shape[-1])

            # # preprocess all data
            # # calculate duration of the audio clip
            # waveform = self.data_list[i]['audio_wave']
            # sample_rate=16000
            # clip_duration = len(waveform) / sample_rate
            # """
            # # for wav2vec2, 1 Token corresponds to ~ 321.89 discrete samples
            # # to get precisely 64 tokens (a hyperparameter that can be changed), the length of input discrete samples to the model should be 321.89 * 64
            # # divide the above by the clip duration to get new sample rate (or) new_sample_rate * clip_duration = 321.89 * num tokens
            # """
            # new_sample_rate = int(321.893491124260 * self.num_av_tokens / clip_duration)
            # # resample
            # waveform = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)
            # # required by collate function
            # mono_waveform = waveform.unsqueeze(0)
            # mono_waveform.type(torch.float32)
            # self.data_list[i]['audio_wave'] = mono_waveform

            # # sample 64 face frames
            # target_frames = np.linspace(0, len(frame_names) - 1, num=self.num_av_tokens)
            # target_frames = np.around(target_frames).astype(
            #     int)  # certain frames may be redundant because of rounding to the nearest integer
            # face_frames = []
            # for i in target_frames:
            #     img = np.asarray(Image.open(file_path + frame_names[i])) / 255.0
            #     face_frames.append(self.transforms(img))
            # face_frames = torch.stack(face_frames, 0)
            # face_frames.type(torch.float32)

        # self.max_audio_wave_frame = max(frame_length)
        self.max_audio_wave_frame = 1000000
        for i in range(len(self.data_list)):
            self.data_list[i]['audio_wave']=torch.nn.functional.pad(self.data_list[i]['audio_wave'], (0, self.max_audio_wave_frame - frame_length[i])) # frame_length[i]=self.data_list[i]['audio_wave'].shape[-1]
            self.data_list[i]['audio_wave']=self.data_list[i]['audio_wave'][:,:self.max_audio_wave_frame]

        self.num_av_tokens = num_av_tokens

    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        # return format --> sample = {'video_x': video_x, 'x_affect': x_affect, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'audio_wave': audio_wave, 'videoname': videoname}

        sample = self.data_list[idx]

        # sample['audio_wave'] = sample['audio_wave'][:,:10000] # 489440

        if self.transform:
            sample = self.transform(sample)
        return sample

class All_test(Dataset):

    def __init__(self, list_dir,  num_av_tokens=64, transform=None):
        with open(list_dir, 'rb') as f:
            self.data_list = pickle.load(f)
        self.transform = transform

        frame_length = []
        for i in range(len(self.data_list)):
            if self.data_list[i]['audio_x'].shape[-1] > 3:
                self.data_list[i]['audio_x']=self.data_list[i]['audio_x'][:, :, :3] # for MDPE
            frame_length.append(self.data_list[i]['audio_wave'].shape[-1])
        # self.max_audio_wave_frame = max(frame_length)
        self.max_audio_wave_frame = 1000000
        for i in range(len(self.data_list)):
            self.data_list[i]['audio_wave']=torch.nn.functional.pad(self.data_list[i]['audio_wave'], (0, self.max_audio_wave_frame - frame_length[i])) # frame_length[i]=self.data_list[i]['audio_wave'].shape[-1]
            self.data_list[i]['audio_wave']=self.data_list[i]['audio_wave'][:,:self.max_audio_wave_frame]

        self.num_av_tokens = num_av_tokens

    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        # return format --> sample = {'video_x': video_x, 'x_affect': x_affect, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'audio_wave': audio_wave, 'videoname': videoname}

        # sample['audio_wave'] = sample['audio_wave'][:,:10000] # 489440
        
        sample = self.data_list[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample
