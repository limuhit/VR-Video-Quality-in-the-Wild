import os
import pdb
import csv
import random

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import scipy.io as scio


class VR_all_images(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, seed=0):
        super(VR_all_images, self).__init__()
        if database_name[:6] == 'VR_all':
            dataInfo = pd.read_csv(filename_path)
            length = dataInfo.shape[0]
            random.seed(seed)
            np.random.seed(seed)
            index_rd = np.random.permutation(length)
            train_index = index_rd[0:int(length * 0.6)]
            val_index = index_rd[int(length * 0.6):int(length * 0.8)]
            test_index = index_rd[int(length * 0.8):]
            testdf = dataInfo.iloc[test_index]
            if database_name == 'VR_all_train':
                self.video_names = dataInfo.iloc[train_index]['VideoName'].tolist()
                self.score = dataInfo.iloc[train_index]['MOS'].tolist()
            elif database_name == 'VR_all_val':
                self.video_names = dataInfo.iloc[val_index]['VideoName'].tolist()
                self.score = dataInfo.iloc[val_index]['MOS'].tolist()
            elif database_name == 'VR_all_7A_test':
                self.video_names = testdf[testdf['VideoName'].str.slice(0, 4) == '7_A_']['VideoName'].tolist()
                self.score = testdf[testdf['VideoName'].str.slice(0, 4) == '7_A_']['MOS'].tolist()
            elif database_name == 'VR_all_7B_test':
                self.video_names = testdf[testdf['VideoName'].str.slice(0, 4) == '7_B_']['VideoName'].tolist()
                self.score = testdf[testdf['VideoName'].str.slice(0, 4) == '7_B_']['MOS'].tolist()
            elif database_name == 'VR_all_15A_test':
                self.video_names = testdf[testdf['VideoName'].str.slice(0, 4) == '15_A']['VideoName'].tolist()
                self.score = testdf[testdf['VideoName'].str.slice(0, 4) == '15_A']['MOS'].tolist()
            elif database_name == 'VR_all_15B_test':
                self.video_names = testdf[testdf['VideoName'].str.slice(0, 4) == '15_B']['VideoName'].tolist()
                self.score = testdf[testdf['VideoName'].str.slice(0, 4) == '15_B']['MOS'].tolist()



        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[:2] == 'VR':
            video_name = self.video_names[idx]
            video_height_crop = self.crop_size
            video_width_crop = self.crop_size * 2

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        video_channel = 3

        if video_name[:2] == '7_':
            video_name_str = video_name[2:]
            path_name = os.path.join(self.videos_dir, video_name_str)
            video_length_read = 7
            transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

            # fix random
            seed = np.random.randint(2023)
            random.seed(seed)
            for i in range(video_length_read):

                imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame

            return transformed_video, video_score, video_name


        elif video_name[:3] == '15_':
            video_name_str = video_name[3:]
            path_name = os.path.join(self.videos_dir, video_name_str)
            video_length_read = 15
            transformed_video = torch.zeros([video_length_read // 2, video_channel, video_height_crop, video_width_crop])

            # fix random
            seed = np.random.randint(2023)
            random.seed(seed)
            for i in range(video_length_read // 2):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(2*i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame

            return transformed_video, video_score, video_name


class Viewport_all_images(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, seed=0):
        super(Viewport_all_images, self).__init__()
        if database_name[:6] == 'VR_all':
            dataInfo = pd.read_csv(filename_path)
            random.seed(seed)
            np.random.seed(seed)
            video = dataInfo['VideoName'].str.split('/', expand=True)[0]
            dataInfo['video'] = video
            unique_video = list(set(video.tolist()))
            random.shuffle(unique_video)
            unique_video_train = unique_video[:int(len(unique_video) * 0.6)]
            unique_video_val = unique_video[int(len(unique_video) * 0.6):int(len(unique_video) * 0.8)]
            unique_video_test = unique_video[int(len(unique_video) * 0.8):]

            mos_train = dataInfo.loc[dataInfo['video'].isin(unique_video_train)]
            mos_val = dataInfo.loc[dataInfo['video'].isin(unique_video_val)]
            mos_test = dataInfo.loc[dataInfo['video'].isin(unique_video_test)]
            if database_name == 'VR_all_train':
                self.video_names = mos_train['VideoName'].tolist()
                self.score = mos_train['MOS'].tolist()
            elif database_name == 'VR_all_val':
                self.video_names = mos_val['VideoName'].tolist()
                self.score = mos_val['MOS'].tolist()
            elif database_name == 'VR_all_7A_test':
                self.video_names = mos_test[mos_test['VideoName'].str.slice(0, 4) == '7_A_']['VideoName'].tolist()
                self.score = mos_test[mos_test['VideoName'].str.slice(0, 4) == '7_A_']['MOS'].tolist()
            elif database_name == 'VR_all_7B_test':
                self.video_names = mos_test[mos_test['VideoName'].str.slice(0, 4) == '7_B_']['VideoName'].tolist()
                self.score = mos_test[mos_test['VideoName'].str.slice(0, 4) == '7_B_']['MOS'].tolist()
            elif database_name == 'VR_all_15A_test':
                self.video_names = mos_test[mos_test['VideoName'].str.slice(0, 4) == '15_A']['VideoName'].tolist()
                self.score = mos_test[mos_test['VideoName'].str.slice(0, 4) == '15_A']['MOS'].tolist()
            elif database_name == 'VR_all_15B_test':
                self.video_names = mos_test[mos_test['VideoName'].str.slice(0, 4) == '15_B']['VideoName'].tolist()
                self.score = mos_test[mos_test['VideoName'].str.slice(0, 4) == '15_B']['MOS'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name[:2] == 'VR':
            video_name = self.video_names[idx]
            video_height_crop = self.crop_size
            video_width_crop = self.crop_size

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        video_channel = 3

        if video_name[:2] == '7_':
            video_name_str = video_name[2:]
            path_name = os.path.join(self.videos_dir, video_name_str)
            video_length_read = 7
            transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

            # fix random
            seed = np.random.randint(2023)
            random.seed(seed)
            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(1 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame

            return transformed_video, video_score, video_name.split('/')[0]


        elif video_name[:3] == '15_':
            video_name_str = video_name[3:]
            path_name = os.path.join(self.videos_dir, video_name_str)
            video_length_read = 15
            transformed_video = torch.zeros([video_length_read // 2, video_channel, video_height_crop, video_width_crop])

            # fix random
            seed = np.random.randint(2023)
            random.seed(seed)
            for i in range(video_length_read // 2):
                imge_name = os.path.join(path_name, '{:03d}'.format(int(2 * i)) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame

            return transformed_video, video_score, video_name.split('/')[0]

