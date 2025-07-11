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

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        if 'DD_label' not in sample.keys():
            sample['DD_label'] = torch.tensor(0,dtype=torch.int64)
        video_x, audio_x, OpenFace_x, DD_label, x_affect, audio_wave, videoname  = sample['video_x'], sample['audio_x'], sample['OpenFace_x'], sample['DD_label'], sample['x_affect'], sample['audio_wave'], sample['videoname']

        new_video_x = (video_x - 127.5)/128     # [-1,1] for resnet [C,T,H,W]
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for vit
        # new_video_x = torch.stack([normalize(frame) for frame in new_video_x]) 

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for vit
        # # 如果是多个 clip: [Clip, C, T, H, W]
        # if video_x.dim() == 5:
        #     video_x = video_x / 255.0  # 归一化到 [0,1]
        #     # 遍历每个 clip 和每一帧： [C, T, H, W] -> [T, C, H, W] -> normalize each frame
        #     new_video_x = torch.stack([
        #         torch.stack([normalize(frame) for frame in clip.permute(1, 0, 2, 3)], dim=0).permute(1, 0, 2, 3)
        #         for clip in video_x
        #     ], dim=0)
        #     # 输出仍是 [Clip, C, T, H, W]
        
        # # 如果是单个 clip: [C, T, H, W]
        # elif video_x.dim() == 4:
        #     video_x = video_x / 255.0
        #     video_x = video_x.permute(1, 0, 2, 3)  # [T, C, H, W]
        #     video_x = torch.stack([normalize(frame) for frame in video_x], dim=0)
        #     new_video_x = video_x.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        new_audio_x = (audio_x - 127.5)/128     # [-1,1] for resnet
        # new_audio_x = audio_x / 255.0
        # new_audio_x = normalize(new_audio_x) 

        new_OpenFace_x = OpenFace_x
        #new_OpenFace_x = (OpenFace_x*255 - 127.5)/128     # [-1,1], OpenFace_x --> Original [0,1]
        return {'video_x': new_video_x, 'audio_x': new_audio_x, 'OpenFace_x': new_OpenFace_x, 'DD_label': DD_label, 'x_affect': x_affect, 'audio_wave': audio_wave, 'videoname': videoname}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        video_x, audio_x, OpenFace_x, DD_label, x_affect, audio_wave, videoname  = sample['video_x'], sample['audio_x'], sample['OpenFace_x'], sample['DD_label'], sample['x_affect'], sample['audio_wave'], sample['videoname']
        
        new_video_x = np.zeros((face_frames, 224, 224, 3))

        p = random.random()
        if p < 0.5:
            for i in range(face_frames):
                # video 
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image

                
            return {'video_x': new_video_x, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'x_affect': x_affect, 'audio_wave': audio_wave, 'videoname': videoname}
        else:
            #print('no Flip')
            return {'video_x': video_x, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'x_affect': x_affect, 'audio_wave': audio_wave, 'videoname': videoname}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample, resize_image_size=224): # 160
        video_x, audio_x, OpenFace_x, DD_label, x_affect, audio_wave, videoname  = sample['video_x'], sample['audio_x'], sample['OpenFace_x'], sample['DD_label'], sample['x_affect'], sample['audio_wave'], sample['videoname']
        
        # swap color axis because
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x C X T x H X W
        if resize_image_size:
            T = video_x.shape[0]
            new_size = (resize_image_size, resize_image_size)
            resized_frames = np.stack([
                cv2.resize(video_x[i], new_size)
                for i in range(T)
            ])
        video_x = resized_frames.transpose((3, 0, 1, 2))# [C,T,H,W] # for resnet
        # video_x = video_x.transpose((1,0,2,3)) # [T,C,H,W] for vit
        # video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
        
        audio_x = audio_x.transpose((2, 0, 1))
        audio_x = np.array(audio_x)
        
        # numpy image: (batch_size) x T x C
        # torch image: (batch_size) x C X T
        OpenFace_x = OpenFace_x.transpose((1, 0))
        OpenFace_x = np.array(OpenFace_x)
        
        x_affect = x_affect.transpose((1, 0))
        x_affect = np.array(x_affect)
                        
        DD_label_np = np.array([0],dtype=np.int64) # np.long
        DD_label_np[0] = DD_label
        
        # return {'video_x': torch.from_numpy(video_x.astype(np.float32)).float(), 'audio_x': torch.from_numpy(audio_x.astype(np.float32)).float(), 'OpenFace_x': torch.from_numpy(OpenFace_x.astype(np.float32)).float(), 'DD_label': torch.from_numpy(DD_label_np.astype(np.long)).long(), 'x_affect': torch.from_numpy(x_affect.astype(np.float32)).float()}
        
        audio_wave = audio_wave.squeeze(0) # for wav2vec

        return {'video_x': torch.from_numpy(video_x.astype(np.float32)).float(), 'audio_x': torch.from_numpy(audio_x.astype(np.float32)).float(), 'OpenFace_x': torch.from_numpy(OpenFace_x.astype(np.float32)).float(), 'DD_label': torch.from_numpy(DD_label_np.astype(np.int64)).long(), 'x_affect': torch.from_numpy(x_affect.astype(np.float32)).float(), 'audio_wave': audio_wave, 'videoname': videoname}


class ToTensor_test(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample, resize_image_size=224): #160
        if 'DD_label' not in sample.keys():
            sample['DD_label'] = torch.tensor(0,dtype=torch.int64)
        video_x, audio_x, OpenFace_x, DD_label, x_affect, audio_wave, videoname  = sample['video_x'], sample['audio_x'], sample['OpenFace_x'], sample['DD_label'], sample['x_affect'], sample['audio_wave'], sample['videoname']
        
        # swap color axis because
        # numpy image: (batch_size) x Clip x T x H x W x C
        # torch image: (batch_size) x Clip x C X T x H X W
        if resize_image_size:
            L, T = video_x.shape[0], video_x.shape[1]
            new_size = (resize_image_size, resize_image_size)
            resized_frames = np.stack([
                [cv2.resize(video_x[j,i], new_size)
                for i in range(T)] for j in range(L)
            ])
        video_x = resized_frames.transpose((0, 4, 1, 2, 3)) # [Clip,C,T,H,W] # for resnet
        # video_x = video_x.transpose((0,2,1,3,4)) # [Clip,T,C,H,W] # for vit
        video_x = np.array(video_x)
        
        audio_x = audio_x.transpose((0, 3, 1, 2))
        audio_x = np.array(audio_x)
        
        # numpy image: (batch_size) x Clip x T x C
        # torch image: (batch_size) x Clip x C X T
        OpenFace_x = OpenFace_x.transpose((0, 2, 1))
        OpenFace_x = np.array(OpenFace_x)
        
        x_affect = x_affect.transpose((0, 2, 1))
        x_affect = np.array(x_affect)
                        
        DD_label_np = np.array([0],dtype=np.int64)
        DD_label_np[0] = DD_label

        audio_wave = audio_wave
        
        return {'video_x': torch.from_numpy(video_x.astype(np.float32)).float(), 'audio_x': torch.from_numpy(audio_x.astype(np.float32)).float(), 'OpenFace_x': torch.from_numpy(OpenFace_x.astype(np.float32)).float(), 'DD_label': torch.from_numpy(DD_label_np.astype(np.int64)).long(), 'x_affect': torch.from_numpy(x_affect.astype(np.float32)).float(), 'audio_wave': audio_wave, 'videoname': videoname}