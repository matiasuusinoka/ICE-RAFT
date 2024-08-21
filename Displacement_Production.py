import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.models.optical_flow import Raft_Large_Weights
from raft_with_context import raft_context_model
from glob import glob
from scipy import interpolate
from multiprocessing import Pool

from utils.displacement_support import preprocess, forward_interpolate, bf_stepping

import os
import argparse

def displacement_production(args):
    
    resolution = args.image_resolution
    iters = args.update_iterations

    filenames = sorted(glob(f'{args.image_path}/*.png'))[::args.displacement_resolution]
    flow_low = None
    
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not args.custom_model:
        model = raft_context_model(weights=Raft_Large_Weights.DEFAULT, progress=False)
    else: 
        model.load_state_dict(torch.load(args.model))
    
    model.to(device)
    model.eval()

    for step in range(1,len(filenames)):
        
        # Context flow intialization
        for reso_num, conscale in enumerate(args.temporal_scales):
            bf_steps = bf_stepping(step, conscale, len(filenames))
            img1_batch = torch.stack([read_image(filenames[bf_steps[0]], mode=torchvision.io.image.ImageReadMode.RGB)])
            img2_batch = torch.stack([read_image(filenames[bf_steps[1]], mode=torchvision.io.image.ImageReadMode.RGB)])
            img1_batch, img2_batch = preprocess(img1_batch, img2_batch, resolution)
            
            init_flow = None if reso_num == 0 else context_flow
            
            context_flow = model(img1_batch.to(device), img2_batch.to(device), flow_init=init_flow, num_flow_updates = iters, return_context=True)
            context_flow = forward_interpolate(context_flow[0]/conscale)[None].to(device)
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

    # This is only for radar data with a shadow sector. With other data sets this can be removed.
    displacements[:,:,500:,500:] = 0
    np.save(f'{args.image_path}/displacements/{args.result_file_name}',displacements)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_model', default=None, help="Using pre-trained custom models.")
    parser.add_argument('--image_path', default='./demo_data/', help="Indicate the path to the images to be analyzed.")
    parser.add_argument('--result_file_name', default='demo_displacements', help="Give a name to the created displacement file.")
    parser.add_argument('--image_resolution', default= 1000, type=int, help="At what resolution will the images be analyzed.")
    parser.add_argument('--displacement_resolution', default= 1, type=int, help="Define temporal resolution for producing the displacements.")
    parser.add_argument('--temporal_scales', default= [6,4], nargs='+', type=int, help="Resolutions for the temporal resolution tree.")
    parser.add_argument('--update_iterations', default= 32, type=int, help="The number of iterations done by each layer in the resolution tree.")
    
    # These can be used with further analysis
    #parser.add_argument('--day_range', nargs='+', type=int, help="Give the starting and ending days of the analyzed period.", required=True)
    #parser.add_argument('--month_and_year', nargs='+', type=int, help="Give the month and year that will be analyzed.", required=True)
    
    args = parser.parse_args()

    for path in [args.image_path, f'{args.image_path}/displacements']: 
        if not os.path.isdir(path): os.mkdir(path)

    #day_start, day_end = args.day_range
    
    # Parallelization is e.g. analysing data simultaneously for multiple days. For demo case this is irrelevant.
    #with Pool(processes=1) as p:
    #    p.map(displacement_production, np.arange(day_start, day_end), args)
    
    displacement_production(args)
