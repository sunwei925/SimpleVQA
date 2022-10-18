# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim

import torch.nn as nn

from data_loader import VideoDataset_images_with_motion_features
from utils import performance_fit
from utils import L1RankLoss



from model import UGC_BVQA_model

from torchvision import transforms
import time


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    if config.model_name == 'UGC_BVQA_model':
        print('The current model is ' + config.model_name)
        model = UGC_BVQA_model.resnet50(pretrained=True)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)
    

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.conv_base_lr, weight_decay = 0.0000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    if config.loss_type == 'L1RankLoss':
        criterion = L1RankLoss()

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))
       

    videos_dir = 'LSVQ_image'
    feature_dir = 'LSVQ_SlowFast2_feature'
    datainfo_train = 'data/LSVQ_whole_train.csv'
    datainfo_test = 'data/LSVQ_whole_test.csv'
    datainfo_test_1080p = 'data/LSVQ_whole_test_1080p.csv'
    transformations_train = transforms.Compose([transforms.Resize(config.resize), transforms.RandomCrop(config.crop_size), transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose([transforms.Resize(config.resize),transforms.CenterCrop(config.crop_size),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        
    trainset = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_train, transformations_train, 'LSVQ_train', config.crop_size, 'SlowFast')
    testset = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_test, transformations_test, 'LSVQ_test', config.crop_size, 'SlowFast')
    testset_1080p = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_test_1080p, transformations_test, 'LSVQ_test_1080p', config.crop_size, 'SlowFast')


    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)
    test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, batch_size=1,
        shuffle=False, num_workers=config.num_workers)


    best_test_criterion = -1  # SROCC min
    best_test = []
    best_test_1080p = []

    print('Starting training:')

    old_save_name = None

    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, feature_3D, mos, _) in enumerate(train_loader):
            

            video = video.to(device)
            feature_3D = feature_3D.to(device)
            labels = mos.to(device).float()
            
            outputs = model(video, feature_3D)
            optimizer.zero_grad()
            
            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()
            
            optimizer.step()

            if (i+1) % (config.print_samples//config.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples//config.train_batch_size)
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
            label = np.zeros([len(testset)])
            y_output = np.zeros([len(testset)])
            for i, (video, feature_3D, mos, _) in enumerate(test_loader):
                
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                label[i] = mos.item()
                outputs = model(video, feature_3D)

                y_output[i] = outputs.item()
            
            test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)
            
            print('Epoch {} completed. The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                test_SRCC, test_KRCC, test_PLCC, test_RMSE))

            label_1080p = np.zeros([len(testset_1080p)])
            y_output_1080p = np.zeros([len(testset_1080p)])
            for i, (video, feature_3D, mos, _) in enumerate(test_loader_1080p):
                
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                label_1080p[i] = mos.item()
                outputs = model(video, feature_3D)

                y_output_1080p[i] = outputs.item()
            
            test_PLCC_1080p, test_SRCC_1080p, test_KRCC_1080p, test_RMSE_1080p = performance_fit(label_1080p, y_output_1080p)
            
            print('Epoch {} completed. The result on the test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                test_SRCC_1080p, test_KRCC_1080p, test_PLCC_1080p, test_RMSE_1080p))
                
            if test_SRCC > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC
                best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                best_test_1080p = [test_SRCC_1080p, test_KRCC_1080p, test_PLCC_1080p, test_RMSE_1080p]
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)

                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)

                save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                    config.database + '_' + config.loss_type + '_NR_v'+ str(config.exp_version) \
                        + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC))
                torch.save(model.state_dict(), save_model_name)
                old_save_name = save_model_name


    print('Training completed.')
    print('The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        best_test[0], best_test[1], best_test[2], best_test[3]))
    print('The best training result on the test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        best_test_1080p[0], best_test_1080p[1], best_test_1080p[2], best_test_1080p[3]))

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default = 0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default = 1000)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=520)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=10)
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='L1RankLoss')
    
    config = parser.parse_args()
    main(config)