import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.models.optical_flow import Raft_Large_Weights
from raft_large import raft_large_model
from glob import glob
from scipy import interpolate
from multiprocessing import Pool

from utils.displacement_support import preprocess, forward_interpolate, bf_stepping

import os
import argparse

def displacement_production(day, args):
    
    month, year = args.month_and_year
    resolution = args.image_resolution
    iters = args.update_iterations

    if day < 10: day = f'0{day}'
    steps = args.displacement_resolution
    filenames = sorted(glob(f'{args.image_path}/{year}{month}{day}*.png'))[::steps]
    flow_low = None
    
    device = 'cuda'
    model = raft_large_model(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model.eval()

    for step in range(1,len(filenames)):

        # Context flow intialization
        # (This could be included into the actual architecture, but due to memory management we produce the context flow initialization here.
        for conscale in args.temporal_scales:
            bf_steps = bf_stepping(step, conscale)
            img1_batch = torch.stack([read_image(filenames[bf_steps[0]], mode=torchvision.io.image.ImageReadMode.RGB)])
            img2_batch = torch.stack([read_image(filenames[bf_steps[1]], mode=torchvision.io.image.ImageReadMode.RGB)])
            img1_batch, img2_batch = preprocess(img1_batch, img2_batch, resolution)
            context_flow = model(img1_batch.to(device), img2_batch.to(device), num_flow_updates = iters, return_context=True)
            context_flow = forward_interpolate(context_flow[0]/conscale)[None].cuda()
            del img1_batch, img2_batch

        # Actual flow estimate
        img1_batch = torch.stack([read_image(filenames[step-1], mode=torchvision.io.image.ImageReadMode.RGB)])
        img2_batch = torch.stack([read_image(filenames[step], mode=torchvision.io.image.ImageReadMode.RGB)])
        img1_batch, img2_batch = preprocess(img1_batch, img2_batch, resolution)
        upsampled_flow = model(img1_batch.to(device), img2_batch.to(device), num_flow_updates = iters, flow_init=context_flow)
        del img1_batch, img2_batch, context_flow
        
        upsampled_flow = upsampled_flow.detach().numpy()
        displacements = upsampled_flow if step == 1 else np.concatenate((displacements,upsampled_flow))
        del upsampled_flow

    displacements[:,:,500:,500:] = 0
    np.save(f'data/displacements/{day}{month}{year}',displacements)
    
if __name__ == '__main__':
    
    for path in ['data','data/displacements']:
        if not os.path.isdir(path): os.mkdir(path)
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--day_range', nargs='+', type=int, help="Give the starting and ending days of the analyzed period.", required=True)
    parser.add_argument('--month_and_year', nargs='+', type=int, help="Give the month and year that will be analyzed.", required=True)
    parser.add_argument('--image_path', help="Indicate the path to the images to be analyzed.")
    parser.add_argument('--image_resolution', default= 1000, type=int, help="At what resolution will the images be analyzed.")
    parser.add_argument('--displacement_resolution', default= 10, type=int, help="Define temporal resolution for producing the displacements.")
    parser.add_argument('--temporal_scales', default= [6,4], nargs='+', type=int, help="Resolutions for the temporal resolution tree.")
    parser.add_argument('--update_iterations', default= 32, type=int, help="The number of iterations done by each layer in the resolution tree.")
    args = parser.parse_args()

    day_start, day_end = args.day_range
    
    with Pool(processes=1) as p:
        p.map(displacement_production, np.arange(day_start, day_end), args)