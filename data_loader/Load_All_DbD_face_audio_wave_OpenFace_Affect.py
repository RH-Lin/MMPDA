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
import bisect
from torch.utils.data import Sampler

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

 # --- 1) Map-style dataset that concatenates domains in order
class All_train(Dataset):
    def __init__(self, list_dir, transform=None, max_wave_frames=1_000_000):
        self.domains = []
        for pkl in list_dir:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            for sample in data:
                # trim to 3 channels
                if sample['audio_x'].shape[-1] > 3:
                    sample['audio_x'] = sample['audio_x'][:, :, :3]
                # pad / truncate waveform
                L = sample['audio_wave'].shape[-1]
                pad = max_wave_frames - L
                sample['audio_wave'] = torch.nn.functional.pad(
                    sample['audio_wave'], (0, pad)
                )[:, :max_wave_frames]

            self.domains.append(data)

        lengths = [len(d) for d in self.domains]
        # cumulative lengths: e.g. [100, 300, 450]
        self.cumlen = list(torch.tensor(lengths).cumsum(0).tolist())
        self.transform = transform

    def __len__(self):
        return self.cumlen[-1]

    def __getitem__(self, idx):
        # figure out which domain
        domain_idx = bisect.bisect_right(self.cumlen, idx)
        if domain_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumlen[domain_idx - 1]

        sample = self.domains[domain_idx][local_idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


# --- 2) Sampler that shuffles within each domain, but yields domains in order
class DomainShuffleSampler(Sampler):
    def __init__(self, dataset: All_train):
        self.dataset = dataset
        # precompute offsets and lengths
        lens = [len(d) for d in dataset.domains]
        offsets = [0]
        for l in lens[:-1]:
            offsets.append(offsets[-1] + l)
        self.offsets = offsets
        self.lengths = lens

    def __iter__(self):
        # for each domain in turn, yield its indices in random order
        for dom_idx, dom_len in enumerate(self.lengths):
            perm = torch.randperm(dom_len).tolist()
            base = self.offsets[dom_idx]
            for i in perm:
                yield base + i

    def __len__(self):
        return len(self.dataset)

class All_test(Dataset):

    def __init__(self, list_dir,  num_av_tokens=64, transform=None):
        with open(list_dir, 'rb') as f:
            self.data_list = pickle.load(f)
        self.transform = transform

        frame_length = []
        for i in range(len(self.data_list)):
            if self.data_list[i]['audio_x'].shape[-1] > 3:
                self.data_list[i]['audio_x']=self.data_list[i]['audio_x'][:, :, :3] # for MDPE and test
            frame_length.append(self.data_list[i]['audio_wave'].shape[-1])
        # self.max_audio_wave_frame = max(frame_length)
        self.max_audio_wave_frame = 1000000
        for i in range(len(self.data_list)):
            self.data_list[i]['audio_wave']=torch.nn.functional.pad(self.data_list[i]['audio_wave'], (0, self.max_audio_wave_frame - frame_length[i])) # frame_length[i]=self.data_list[i]['audio_wave'].shape[-1]
            self.data_list[i]['audio_wave']=self.data_list[i]['audio_wave'][:,:1000000]

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
