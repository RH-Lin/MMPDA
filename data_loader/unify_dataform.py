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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import librosa

def softmax_np(x):
    # x: numpy array of shape (N, D)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)

path_list_dir=[
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features/BagOfLies/BgOL_features.pkl",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features/MU3D/MU3D_features.pkl",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features/RLT/RLT_features.pkl",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features/MMDD_stage2_features/DOLOS_train_l464_t365.pkl",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features/MMDD_stage2_features/MDPE_train_balanced_l493_t492.pkl",

    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features/BoxOfLies/BOL_test_features.pkl",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features/MMDD2025_final_test_features.pkl"
    ]

after_processed_dir=[
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features_unify/BagOfLies/",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features_unify/MU3D/",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features_unify/RLT/",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features_unify/MMDD_stage2_features/",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features_unify/MMDD_stage2_features/",

    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features_unify/BoxOfLies/",
    "/presearch_lin/DeceptionDetection/Deception_datasets/MMDD2025_features_unify/"
]

for path in after_processed_dir:
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

data_list = []

for i, path in enumerate(path_list_dir):
    dataset_name = os.path.basename(path).split(".pkl")[0]
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print(1)

    # ============= unify affect 5 emotion ============
    if dataset_name in ('DOLOS_train_l464_t365', 'MDPE_train_balanced_l493_t492', 'MMDD2025_final_test_features'):
        for sample in data:
            if len(sample['x_affect'].shape) > 2:
                sample['x_affect'][0][:,:5] = softmax_np(sample['x_affect'][0][:,:5])
            else:
                sample['x_affect'][:,:5] = softmax_np(sample['x_affect'][:,:5])

    # ============= unify video face RGB ============
    if dataset_name in ('DOLOS_train_l464_t365', 'MDPE_train_balanced_l493_t492', 'MMDD2025_final_test_features'):
        for sample in data:

            if len(sample['video_x'].shape) > 4:
                org_img=sample['video_x'].squeeze() # for MDPE and test
            else:
                org_img=sample['video_x']

            rgb_face = np.array([cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2RGB) for frame in org_img])

            if len(sample['video_x'].shape) > 4:
                sample['video_x']=np.expand_dims(rgb_face, axis=0).astype(np.uint8) # for test
            else:
                sample['video_x']=rgb_face.astype(np.uint8)

    # ============= unify audio mel spectrum ============
    if dataset_name in ('DOLOS_train_l464_t365', 'MDPE_train_balanced_l493_t492', 'MMDD2025_final_test_features'):
        for sample in data:
            
            if len(sample['audio_x'].shape) > 3:
                org_img=sample['audio_x'].squeeze() # for MDPE and test
            else:
                org_img=sample['audio_x']
            
            gray_mel = org_img.transpose((2, 0, 1))[0]
            gray_mel = (gray_mel - gray_mel.min()) / (gray_mel.max() - gray_mel.min())
            colormap = plt.colormaps.get_cmap('viridis') # default for ax.imshow
            colored_mel = colormap(gray_mel)[:, :, :3] * 255

            if len(sample['audio_x'].shape) > 3:
                sample['audio_x']=np.expand_dims(colored_mel, axis=0).astype(np.uint8) # for test
            else:
                sample['audio_x']=colored_mel.astype(np.uint8)
            

    save_filename = after_processed_dir[i] + dataset_name + '.pkl'
    with open(save_filename, 'wb') as f:
        pickle.dump(data, f)