import pdb
import numpy as np
import os
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import torch
from PCONV_operator import Rotate
import numpy as np


def tensor2img(x):
    tx = x.to('cpu').detach().numpy().transpose(1,2,0)
    tx[tx<0]=0
    tx[tx>255]=255
    tx = tx.astype(np.uint8)
    return tx


def extract_frame(videos_dir, video_name, lon, lat, save_folder):
    vp = Rotate().to('cuda:0')
    try:
        filename = os.path.join(videos_dir, video_name)
        video_name_str = str(video_name)
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

        video_height_resize = 512
        video_width_resize = 1024
        print(video_height_resize)
        print(video_width_resize)

    except:
        print(filename)

    else:
        dim = (video_width_resize, video_height_resize)

        video_read_index = 0

        frame_idx = 0

        video_length_min = 15

        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == 0):
                    read_frame = cv2.resize(frame, dim)
                    data = torch.from_numpy(read_frame.transpose(2, 0, 1).astype(np.float32)).view(1, 3, video_height_resize, video_width_resize).contiguous().to('cuda:0')
                    tf = torch.Tensor([lon, lat]).to('cuda:0').contiguous().view(1, 2)  # lon lat
                    y = vp(data, tf)
                    dst = tensor2img(y[0])
                    cv2.imwrite(os.path.join(save_folder, video_name_str,
                                             '{:03d}'.format(video_read_index) + '.png'), dst)
                    video_read_index += 1
                frame_idx += 1

        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(i) + '.png'), dst)

        print('video_read_index:', video_read_index)

        return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return


videos_dir = './VRVideo_Dataset/Video'
info_path = 'data/infoB.csv'
save_folder = './ERP_frame/erp_fp1_512_spAB'

dataInfo = pd.read_csv(info_path)
n_video = dataInfo.shape[0]
video_names = []
HM_pitchs = []
HM_yaws = []

for i in range(n_video):
    video_names.append(dataInfo['VideoName'][i])
    HM_pitchs.append(dataInfo['HM_pitch'][i])
    HM_yaws.append(dataInfo['HM_yaw'][i])


for i in range(n_video):
    video_name = video_names[i]
    lat = HM_pitchs[i]
    lon = HM_yaws[i]
    print('lat:', lat, 'lon', lon)
    print('start extract {}th video: {}'.format(i, video_name))
    extract_frame(videos_dir, video_name, lon*(-1), lat, save_folder)
