import pdb
import os
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import eq2cm

pi = np.pi


def mc2img(x):
    tx = x#.to('cpu').detach().numpy().transpose(1,2,0)
    tx[tx<0]=0
    tx[tx>255]=255
    tx = tx.astype(np.uint8)
    return tx


def extract_frame(videos_dir, video_name, subject, lon, lat, save_folder, info):
    try:
        filename = os.path.join(videos_dir, video_name)
        if info == 'data/scanpathA.csv':
            video_name_str = 'A_' + str(video_name)
        elif info == 'data/scanpathB.csv':
            video_name_str = 'B_' + str(video_name)
        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        print(filename)
        print(video_length)
        print(video_frame_rate)

        exit_folder(os.path.join(save_folder, video_name_str))

        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the heigh of frames
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames
        print(video_height)
        print(video_width)


    except:
        print(filename)

    else:
        # dim = (video_width_resize, video_height_resize)

        video_read_index = 0

        frame_idx = 0

        video_length_min = 15

        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == 0):
                    imgviewport = eq2cm.eq_to_pers(frame, np.pi / 2, np.deg2rad(lon[video_read_index]), np.deg2rad(lat[video_read_index]), 224, 224) # FOV lon lat

                    imgviewport = mc2img(imgviewport)

                    exit_folder(os.path.join(save_folder, video_name_str, str('sub')+str(subject)))
                    cv2.imwrite(os.path.join(save_folder, video_name_str, str('sub')+str(subject),
                                             '{:03d}'.format(video_read_index) + '.png'), imgviewport)

                    video_read_index += 1
                frame_idx += 1

        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str, str('sub')+str(subject),
                                         '{:03d}'.format(i) + '.png'), imgviewport)

        print('video_read_index:', video_read_index)

        return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return


videos_dir = './VRVideo_Dataset/Video'
info_path = 'data/scanpathB.csv'
save_folder = './ERP_frame/viewport_224_fp1_spAB'

dataInfo = pd.read_csv(info_path)

video_names = dataInfo["VideoName"].unique().tolist()
n_video = len(video_names)


for i in range(n_video):
    video_name = video_names[i]
    subjects = dataInfo[dataInfo["VideoName"] == video_name]['subject'].unique().tolist()
    for sub in subjects:
        tmp = dataInfo[(dataInfo["VideoName"] == video_name)&(dataInfo["subject"] == sub)]
        lat = tmp["HM_pitch"].tolist()
        lon = tmp["HM_yaw"].tolist()
        extract_frame(videos_dir, video_name, sub, lon, lat, save_folder, info=info_path)
