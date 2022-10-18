# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from data_loader import VideoDataset_NR_SlowFast_feature

from pytorchvideo.models.hub import slowfast_r50
from torchvision import transforms

def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list

class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)
        

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)

            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
            
        return slow_feature, fast_feature

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = slowfast()
    

    model = model.to(device)

    resize = config.resize
        
    ## training data
    if config.database == 'KoNViD-1k':
        videos_dir = 'konvid1k'
        datainfo_test = 'data/KoNViD-1k_data.mat'

        transformations_test = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])
    
        trainset = VideoDataset_NR_SlowFast_feature(videos_dir, datainfo_test, transformations_test, resize, 'KoNViD-1k')


    elif config.database == 'youtube_ugc':
        videos_dir = 'youtube_ugc/h264'
        datainfo_test = 'data/youtube_ugc_data.mat'

        transformations_test = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])
    
        trainset = VideoDataset_NR_SlowFast_feature(videos_dir, datainfo_test, transformations_test, resize, 'youtube_ugc')


    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)


    # do validation after each epoch
    with torch.no_grad():
        model.eval()

        for i, (video, video_name) in enumerate(train_loader):
            video_name = video_name[0]
            print(video_name)
            if not os.path.exists(config.feature_save_folder + video_name):
                os.makedirs(config.feature_save_folder + video_name)
            
            for idx, ele in enumerate(video):
                # ele = ele.to(device)
                ele = ele.permute(0, 2, 1, 3, 4)             
                inputs = pack_pathway_output(ele, device)
                slow_feature, fast_feature = model(inputs)
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_slow_feature', slow_feature.to('cpu').numpy())
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_fast_feature', fast_feature.to('cpu').numpy())







        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--resize', type=int, default=112)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str, default='ckpts')

    config = parser.parse_args()

    main(config)


