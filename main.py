from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import json
import argparse
from tqdm import trange

from utils import load_data
from model import init_models, train

seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Trainer facility for ENeRF')
    parser.add_argument('config', help='Path to the config file (default: configs/enerf.json)',
                        default='configs/enerf.json')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as fin:
        config = json.load(fin)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    images, poses, focal = load_data(config['data_path'])
    height, width = images.shape[1:3]
    testimg, testpose = images[config["testimg_idx"]], poses[config["testimg_idx"]]

    images = torch.from_numpy(images).to(device)
    poses = torch.from_numpy(poses).to(device)
    focal = torch.from_numpy(focal).to(device)
    testimg = torch.from_numpy(testimg).to(device)
    testpose = torch.from_numpy(testpose).to(device)

    # Run training session(s)
    for _ in range(config['n_restarts']):
        model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models(config)
        success, train_psnrs, val_psnrs = train(model, fine_model, optimizer, warmup_stopper,
                                                images, poses, focal,
                                                testimg, testpose,
                                                encode, encode_viewdirs,
                                                config)
        if success and val_psnrs[-1] >= config['warmup_min_fitness']:
            print('Training successful!')
            break

    print('')
    print(f'Done!')
