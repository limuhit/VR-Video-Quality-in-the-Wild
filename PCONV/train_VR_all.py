# -*- coding: utf-8 -*-
import argparse
import os
import pdb

import numpy as np
import torch
import torch.optim as optim

import torch.nn as nn
import random
from data_loader import VR_all_images
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss
from model import PCONV_RN50, PCONV_RN50_IQA, PCONV_lightRN, PCONV_lightRN_IQA, RN50, RN50_IQA, RN18, RN18_IQA, RN34, RN34_IQA
from torchvision import transforms
import time


def main(config):
    test_SRCC_7A_list, test_KRCC_7A_list, test_PLCC_7A_list, test_RMSE_7A_list = [], [], [], []
    test_SRCC_7B_list, test_KRCC_7B_list, test_PLCC_7B_list, test_RMSE_7B_list = [], [], [], []
    test_SRCC_15A_list, test_KRCC_15A_list, test_PLCC_15A_list, test_RMSE_15A_list = [], [], [], []
    test_SRCC_15B_list, test_KRCC_15B_list, test_PLCC_15B_list, test_RMSE_15B_list = [], [], [], []
    test_SRCC_7AB_list, test_KRCC_7AB_list, test_PLCC_7AB_list, test_RMSE_7AB_list = [], [], [], []
    test_SRCC_15AB_list, test_KRCC_15AB_list, test_PLCC_15AB_list, test_RMSE_15AB_list = [], [], [], []
    test_SRCC_all_list, test_KRCC_all_list, test_PLCC_all_list, test_RMSE_all_list = [], [], [], []

    for i in range(10):
        config.exp_version = i
        print('%d round training starts here' % i)
        seed = i * 1

        device_id = 0
        device = f'cuda:{device_id}'
        if config.model_name == 'RN18':
            print('The current model is ' + config.model_name)
        elif config.model_name == 'PCONV_RN18':
            print('The current model is ' + config.model_name)
            model = PCONV_lightRN.ResNet18('/data0/wen/resnet/resnet18-f37072fd.pth')
        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)

        if config.trained_model is not None:
            # load the trained model
            print('loading the pretrained model')
            model.load_state_dict(torch.load(config.trained_model))

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
        if config.loss_type == 'plcc':
            criterion = plcc_loss

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))

        transformations_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        ## training data
        if config.database == 'VR_all':
            videos_dir = './ERP_frame/erp_fps1_512_spAB/'
            datainfo_train = 'data/infoAB_all.csv'
            datainfo_val = 'data/infoAB_all.csv'
            datainfo_test = 'data/infoAB_all.csv'

            trainset = VR_all_images(videos_dir, datainfo_train, transformations_train, 'VR_all_train',
                                           config.crop_size, seed)
            valset = VR_all_images(videos_dir, datainfo_val, transformations_train, 'VR_all_val', config.crop_size, seed)
            testset_7A = VR_all_images(videos_dir, datainfo_test, transformations_test, 'VR_all_7A_test', config.crop_size, seed)
            testset_7B = VR_all_images(videos_dir, datainfo_test, transformations_test, 'VR_all_7B_test', config.crop_size, seed)
            testset_15A = VR_all_images(videos_dir, datainfo_test, transformations_test, 'VR_all_15A_test', config.crop_size, seed)
            testset_15B = VR_all_images(videos_dir, datainfo_test, transformations_test, 'VR_all_15B_test', config.crop_size, seed)

        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, drop_last=True,
                                                   shuffle=True, num_workers=config.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.train_batch_size, drop_last=True,
                                                 shuffle=False, num_workers=config.num_workers)
        test_loader_7A = torch.utils.data.DataLoader(testset_7A, batch_size=config.train_batch_size, drop_last=True,
                                                     shuffle=False, num_workers=config.num_workers)
        test_loader_7B = torch.utils.data.DataLoader(testset_7B, batch_size=config.train_batch_size, drop_last=True,
                                                     shuffle=False, num_workers=config.num_workers)
        test_loader_15A = torch.utils.data.DataLoader(testset_15A, batch_size=config.train_batch_size, drop_last=True,
                                                      shuffle=False, num_workers=config.num_workers)
        test_loader_15B = torch.utils.data.DataLoader(testset_15B, batch_size=config.train_batch_size, drop_last=True,
                                                      shuffle=False, num_workers=config.num_workers)

        best_val_criterion = -1  # SROCC min
        best_val = []

        print('Starting training:')


        for epoch in range(config.epochs):
            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (video, mos, _) in enumerate(train_loader):

                video = video.to(device)
                labels = mos.to(device).float()

                outputs = model(video)
                optimizer.zero_grad()

                loss = criterion(outputs, labels)
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
                loss.backward()

                optimizer.step()
                if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                          (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                           avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                    session_start_time = time.time()
            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))

            # do validation after each epoch
            with torch.no_grad():
                model.eval()
                label = []
                y_output = []
                for i, (video, mos, _) in enumerate(val_loader):
                    video = video.to(device)
                    tmp_label = mos.tolist()
                    label = label + tmp_label
                    outputs = model(video)
                    tmp_y_output = outputs.tolist()
                    y_output = y_output + tmp_y_output

                val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(np.array(label), np.array(y_output))

                print(
                    'Epoch {} completed. The result on the validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, val_SRCC, val_KRCC, val_PLCC, val_RMSE))

                label_7A = []
                y_output_7A = []
                for i, (video, mos, _) in enumerate(test_loader_7A):
                    video = video.to(device)
                    tmp_label = mos.tolist()
                    label_7A = label_7A + tmp_label
                    outputs = model(video)
                    tmp_y_output = outputs.tolist()
                    y_output_7A = y_output_7A + tmp_y_output

                test_PLCC_7A, test_SRCC_7A, test_KRCC_7A, test_RMSE_7A = performance_fit(label_7A, y_output_7A)

                print(
                    'Epoch {} completed. The result on the test 7A databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_7A, test_KRCC_7A, test_PLCC_7A, test_RMSE_7A))

                label_7B = []
                y_output_7B = []
                for i, (video, mos, _) in enumerate(test_loader_7B):
                    video = video.to(device)
                    tmp_label = mos.tolist()
                    label_7B = label_7B + tmp_label
                    outputs = model(video)
                    tmp_y_output = outputs.tolist()
                    y_output_7B = y_output_7B + tmp_y_output

                test_PLCC_7B, test_SRCC_7B, test_KRCC_7B, test_RMSE_7B = performance_fit(label_7B, y_output_7B)

                print(
                    'Epoch {} completed. The result on the test 7B databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_7B, test_KRCC_7B, test_PLCC_7B, test_RMSE_7B))

                label_15A = []
                y_output_15A = []
                for i, (video, mos, _) in enumerate(test_loader_15A):
                    video = video.to(device)
                    tmp_label = mos.tolist()
                    label_15A = label_15A + tmp_label
                    outputs = model(video)
                    tmp_y_output = outputs.tolist()
                    y_output_15A = y_output_15A + tmp_y_output

                test_PLCC_15A, test_SRCC_15A, test_KRCC_15A, test_RMSE_15A = performance_fit(label_15A, y_output_15A)

                print(
                    'Epoch {} completed. The result on the test 15A databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_15A, test_KRCC_15A, test_PLCC_15A, test_RMSE_15A))

                label_15B = []
                y_output_15B = []
                for i, (video, mos, _) in enumerate(test_loader_15B):
                    video = video.to(device)
                    tmp_label = mos.tolist()
                    label_15B = label_15B + tmp_label
                    outputs = model(video)
                    tmp_y_output = outputs.tolist()
                    y_output_15B = y_output_15B + tmp_y_output

                test_PLCC_15B, test_SRCC_15B, test_KRCC_15B, test_RMSE_15B = performance_fit(label_15B, y_output_15B)

                print(
                    'Epoch {} completed. The result on the test 15B databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_15B, test_KRCC_15B, test_PLCC_15B, test_RMSE_15B))

                label_7AB = label_7A + label_7B
                y_output_7AB = y_output_7A + y_output_7B
                test_PLCC_7AB, test_SRCC_7AB, test_KRCC_7AB, test_RMSE_7AB = performance_fit(label_7AB, y_output_7AB)

                print(
                    'Epoch {} completed. The result on the test 7AB databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_7AB, test_KRCC_7AB, test_PLCC_7AB, test_RMSE_7AB))

                label_15AB = label_15A + label_15B
                y_output_15AB = y_output_15A + y_output_15B
                test_PLCC_15AB, test_SRCC_15AB, test_KRCC_15AB, test_RMSE_15AB = performance_fit(label_15AB, y_output_15AB)
                print(
                    'Epoch {} completed. The result on the test 15AB databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_15AB, test_KRCC_15AB, test_PLCC_15AB, test_RMSE_15AB))

                label_all = label_7AB + label_15AB
                y_output_all = y_output_7AB + y_output_15AB
                test_PLCC_all, test_SRCC_all, test_KRCC_all, test_RMSE_all = performance_fit(label_all, y_output_all)
                print(
                    'Epoch {} completed. The result on the test all databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_all, test_KRCC_all, test_PLCC_all, test_RMSE_all))


                if val_SRCC > best_val_criterion:
                    print("Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                    best_val_criterion = val_SRCC
                    best_val = [val_SRCC, val_KRCC, val_PLCC, val_RMSE]
                    best_test_7A = [test_SRCC_7A, test_KRCC_7A, test_PLCC_7A, test_RMSE_7A]
                    best_test_7B = [test_SRCC_7B, test_KRCC_7B, test_PLCC_7B, test_RMSE_7B]
                    best_test_15A = [test_SRCC_15A, test_KRCC_15A, test_PLCC_15A, test_RMSE_15A]
                    best_test_15B = [test_SRCC_15B, test_KRCC_15B, test_PLCC_15B, test_RMSE_15B]
                    best_test_7AB = [test_SRCC_7AB, test_KRCC_7AB, test_PLCC_7AB, test_RMSE_7AB]
                    best_test_15AB = [test_SRCC_15AB, test_KRCC_15AB, test_PLCC_15AB, test_RMSE_15AB]
                    best_test_all = [test_SRCC_all, test_KRCC_all, test_PLCC_all, test_RMSE_all]

                    


        print('Training completed.')
        print('The best training result on the validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_val[0], best_val[1], best_val[2], best_val[3]))
        print('The best training result on the test 7A dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_7A[0], best_test_7A[1], best_test_7A[2], best_test_7A[3]))
        print(
            'The best training result on the test 7B dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_7B[0], best_test_7B[1], best_test_7B[2], best_test_7B[3]))
        print(
            'The best training result on the test 15A dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_15A[0], best_test_15A[1], best_test_15A[2], best_test_15A[3]))
        print(
            'The best training result on the test 15B dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_15B[0], best_test_15B[1], best_test_15B[2], best_test_15B[3]))
        print(
            'The best training result on the test 7AB dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_7AB[0], best_test_7AB[1], best_test_7AB[2], best_test_7AB[3]))
        print(
            'The best training result on the test 15AB dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_15AB[0], best_test_15AB[1], best_test_15AB[2], best_test_15AB[3]))
        print(
            'The best training result on the test all dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_all[0], best_test_all[1], best_test_all[2], best_test_all[3]))

        test_SRCC_7A_list.append(best_test_7A[0])
        test_KRCC_7A_list.append(best_test_7A[1])
        test_PLCC_7A_list.append(best_test_7A[2])
        test_RMSE_7A_list.append(best_test_7A[3])

        test_SRCC_7B_list.append(best_test_7B[0])
        test_KRCC_7B_list.append(best_test_7B[1])
        test_PLCC_7B_list.append(best_test_7B[2])
        test_RMSE_7B_list.append(best_test_7B[3])

        test_SRCC_15A_list.append(best_test_15A[0])
        test_KRCC_15A_list.append(best_test_15A[1])
        test_PLCC_15A_list.append(best_test_15A[2])
        test_RMSE_15A_list.append(best_test_15A[3])

        test_SRCC_15B_list.append(best_test_15B[0])
        test_KRCC_15B_list.append(best_test_15B[1])
        test_PLCC_15B_list.append(best_test_15B[2])
        test_RMSE_15B_list.append(best_test_15B[3])

        test_SRCC_7AB_list.append(best_test_7AB[0])
        test_KRCC_7AB_list.append(best_test_7AB[1])
        test_PLCC_7AB_list.append(best_test_7AB[2])
        test_RMSE_7AB_list.append(best_test_7AB[3])

        test_SRCC_15AB_list.append(best_test_15AB[0])
        test_KRCC_15AB_list.append(best_test_15AB[1])
        test_PLCC_15AB_list.append(best_test_15AB[2])
        test_RMSE_15AB_list.append(best_test_15AB[3])

        test_SRCC_all_list.append(best_test_all[0])
        test_KRCC_all_list.append(best_test_all[1])
        test_PLCC_all_list.append(best_test_all[2])
        test_RMSE_all_list.append(best_test_all[3])

    print(
        'The avg 7A results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(test_SRCC_7A_list), np.mean(test_KRCC_7A_list), np.mean(test_PLCC_7A_list), np.mean(test_RMSE_7A_list)))
    print(
        'The std 7A results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(test_SRCC_7A_list), np.std(test_KRCC_7A_list), np.std(test_PLCC_7A_list), np.std(test_RMSE_7A_list)))
    print(
        'The median 7A results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(test_SRCC_7A_list), np.median(test_KRCC_7A_list), np.median(test_PLCC_7A_list), np.median(test_RMSE_7A_list)))

    print(
        'The avg 7B results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(test_SRCC_7B_list), np.mean(test_KRCC_7B_list), np.mean(test_PLCC_7B_list), np.mean(test_RMSE_7B_list)))
    print(
        'The std 7B results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(test_SRCC_7B_list), np.std(test_KRCC_7B_list), np.std(test_PLCC_7B_list), np.std(test_RMSE_7B_list)))
    print(
        'The median 7B results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(test_SRCC_7B_list), np.median(test_KRCC_7B_list), np.median(test_PLCC_7B_list), np.median(test_RMSE_7B_list)))

    print(
        'The avg 15A results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(test_SRCC_15A_list), np.mean(test_KRCC_15A_list), np.mean(test_PLCC_15A_list), np.mean(test_RMSE_15A_list)))
    print(
        'The std 15A results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(test_SRCC_15A_list), np.std(test_KRCC_15A_list), np.std(test_PLCC_15A_list), np.std(test_RMSE_15A_list)))
    print(
        'The median 15A results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(test_SRCC_15A_list), np.median(test_KRCC_15A_list), np.median(test_PLCC_15A_list), np.median(test_RMSE_15A_list)))

    print(
        'The avg 15B results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(test_SRCC_15B_list), np.mean(test_KRCC_15B_list), np.mean(test_PLCC_15B_list), np.mean(test_RMSE_15B_list)))
    print(
        'The std 15B results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(test_SRCC_15B_list), np.std(test_KRCC_15B_list), np.std(test_PLCC_15B_list), np.std(test_RMSE_15B_list)))
    print(
        'The median 15B results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(test_SRCC_15B_list), np.median(test_KRCC_15B_list), np.median(test_PLCC_15B_list), np.median(test_RMSE_15B_list)))

    print(
        'The avg 7AB results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(test_SRCC_7AB_list), np.mean(test_KRCC_7AB_list), np.mean(test_PLCC_7AB_list), np.mean(test_RMSE_7AB_list)))
    print(
        'The std 7AB results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(test_SRCC_7AB_list), np.std(test_KRCC_7AB_list), np.std(test_PLCC_7AB_list), np.std(test_RMSE_7AB_list)))
    print(
        'The median 7AB results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(test_SRCC_7AB_list), np.median(test_KRCC_7AB_list), np.median(test_PLCC_7AB_list), np.median(test_RMSE_7AB_list)))

    print(
        'The avg 15AB results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(test_SRCC_15AB_list), np.mean(test_KRCC_15AB_list), np.mean(test_PLCC_15AB_list), np.mean(test_RMSE_15AB_list)))
    print(
        'The std 15AB results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(test_SRCC_15AB_list), np.std(test_KRCC_15AB_list), np.std(test_PLCC_15AB_list), np.std(test_RMSE_15AB_list)))
    print(
        'The median 15AB results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(test_SRCC_15AB_list), np.median(test_KRCC_15AB_list), np.median(test_PLCC_15AB_list), np.median(test_RMSE_15AB_list)))

    print(
        'The avg all results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(test_SRCC_all_list), np.mean(test_KRCC_all_list), np.mean(test_PLCC_all_list), np.mean(test_RMSE_all_list)))
    print(
        'The std all results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(test_SRCC_all_list), np.std(test_KRCC_all_list), np.std(test_PLCC_all_list), np.std(test_RMSE_all_list)))
    print(
        'The median all results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(test_SRCC_all_list), np.median(test_KRCC_all_list), np.median(test_PLCC_all_list), np.median(test_RMSE_all_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)

    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int, default=0)
    parser.add_argument('--print_samples', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='plcc')

    parser.add_argument('--trained_model', type=str, default=None)

    config = parser.parse_args()

    torch.manual_seed(0)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    torch.utils.backcompat.broadcast_warning.enabled = True

    main(config)
