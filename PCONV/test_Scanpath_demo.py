import argparse
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
from model import RN18, PCONV_lightRN
import cv2
from PIL import Image
import time
from torchvision import transforms
import numpy as np

import erp_rotate.eq2cm as eq2cm

pi = np.pi


def mc2img(x):
    tx = x#.to('cpu').detach().numpy().transpose(1,2,0)
    tx[tx<0]=0
    tx[tx>255]=255
    tx = tx.astype(np.uint8)
    return tx


def video_processing_spatial(dist):
    video_name = dist
    video_name_dis = video_name

    video_capture = cv2.VideoCapture()
    video_capture.open(video_name)
    cap = cv2.VideoCapture(video_name)

    video_channel = 3

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    video_length_read = int(video_length / (2*video_frame_rate))

    transformations = transforms.Compose([transforms.ToTensor(), \
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transformed_video = torch.zeros([video_length_read, video_channel, 224, 224])

    video_read_index = 0
    frame_idx = 0


    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:

            # key frame
            if (video_read_index < video_length_read) and (frame_idx % (2*video_frame_rate) == 0):
                imgviewport = eq2cm.eq_to_pers(frame, np.pi / 2, np.deg2rad(0),
                                               np.deg2rad(0), 224, 224)  # FOV lon lat

                imgviewport = mc2img(imgviewport)
                read_frame = transformations(imgviewport)
                transformed_video[video_read_index] = read_frame
                video_read_index += 1

            frame_idx += 1

    if video_read_index < video_length_read:
        for i in range(video_read_index, video_length_read):
            transformed_video[i] = transformed_video[video_read_index - 1]

    video_capture.release()

    video = transformed_video

    return video, video_name_dis


def main(config):
    device_id = 0
    device = f'cuda:{device_id}'

    model = RN18.ResNet18("https://download.pytorch.org/models/resnet18-f37072fd.pth")

    model = model.to(device)


    if config.method_name == 'single-scale':

        video_dist_spatial, video_name = video_processing_spatial(config.dist)

        with torch.no_grad():
            model.eval()
            video_dist_spatial = video_dist_spatial.to(device)
            video_dist_spatial = video_dist_spatial.unsqueeze(dim=0)

            outputs = model(video_dist_spatial)

            y_val = outputs.item()

            print('The video name: ' + video_name)
            print('The quality socre: {:.4f}'.format(y_val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Scanpath-VQA with single scanpath.')

    # input parameters
    parser.add_argument('--method_name', type=str, default='single-scale')
    parser.add_argument('--dist', type=str, default='./test_VR.mp4')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--is_gpu', action='store_false')

    config = parser.parse_args()

    main(config)