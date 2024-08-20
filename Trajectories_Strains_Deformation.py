import torch
import numpy as np
from glob import glob
from natsort import natsorted
from utils.trajectory_form import form_trajectories
from utils.straintensor_support import kinematic_matrix, DeformationGradient_quadrilateral, infinit_rot
from multiprocessing import Pool

import os
import argparse

def end_to_end(day, args):

    month, year = args.month_and_year
    m = args.spatial_scale
    B = kinematic_matrix(m)
    
    if day < 10: day = f'0{day}'
    displacements = np.load(f'data/displacements/{day}{month}{year}.npy')
    trajectories = form_trajectories(displacements)
    del displacements
    if 'trajectories' in args.saved_quantities: 
        np.save(f'data/trajectories/trajectories_{day}{month}{year}.npy', trajectories)
        
    # Remove the Eulerian displacements to save memory
    if args.remove_displacements == True: os.remove(f'data/displacements/{day}{month}{year}.npy')
    
    if any(metric in args.saved_quantities for metric in ['strain_tensors','deformations']:

        edge_lim = 60
        trajectories = trajectories[:,:,edge_lim:-edge_lim,edge_lim:-edge_lim]
        resolution = trajectories.shape[2]
        trajectories[:,:,int(resolution/2):,int(resolution/2):] = np.nan

        for t in range(0, trajectories.shape[0]):

            u = (trajectories[t]-trajectories[0])*args.pixel_to_metric
            F = np.array([[DeformationGradient_quadrilateral(u[:,x:x+m,y:y+m], B) for y in range(0,u.shape[1]-(m-1),m-1)] for x in range(0,u.shape[2]-(m-1),m-1)])
            del u

            epsilon = tensor_formulation(F, rotation_fixed = args.use_finite_strain)
            del F

            if t == 0: epsilonfull = np.expand_dims(epsilon, axis=0)
            else: epsilonfull = np.concatenate((epsilonfull,np.expand_dims(epsilon, axis=0)))
            del epsilon

        del trajectories

        # Save the strain tensor
        if 'strain_tensors' in args.saved_quantities: 
            np.save(f'data/strain_tensors/epsilon_{day}{month}{year}',epsilonfull)


        if 'deformations' in args.saved_quantities:

            epsilon = np.diff(epsilonfull,axis=0)

            div = np.trace(epsilonfull, axis1=3, axis2=4)
            she = np.sqrt((epsilonfull[:,:,:,0,0] - epsilonfull[:,:,:,1,1])**2 + 4*epsilonfull[:,:,:,0,1]**2)
            tot = np.sqrt(div**2 + she**2)
            data = np.stack((div,she,tot))
            del epsilon, div, she, tot

            np.save(f'data/deformations/deformations_{day}{month}{year}', data)
           
           
if __name__ == '__main__':
    
    for path in ['data','data/displacements']:
        if not os.path.isdir(path): print(f'{path} not found. Produce displacements before formulating trajectories.')

    for path in ['data/trajectories','data/strain_tensors','data/deformations']:
        if not os.path.isdir(path): os.mkdir(path)
           
    parser = argparse.ArgumentParser()
    parser.add_argument('--day_range', nargs='+', type=int, help="Give the starting and ending days of the analyzed period.", required=True)
    parser.add_argument('--month_and_year', nargs='+', type=int, help="Give the month and year that will be analyzed.", required=True)
    parser.add_argument('--spatial_scale', default = 2, type=int, help="Spatial scale defines the size of the deformations objects. m=2 is pixel scale, m=2 the scale of two pixels etc.")
    parser.add_argument('--pixel_to_metric', default = 10, type=int, help="Define the spatial scale of one pixel i.e. how many meters does one pixel represent.")
    parser.add_argument('--use_finite_strain', default = True, help='The deformations will either use infinitesimal or finite strain.')
    parser.add_argument('--saved_quantities', default=['trajectories', 'strain_tensors', 'deformations'], nargs='+', help="Define which quantities will be saved. Options: trajectories, strain_tensors and deformations")
    parser.add_argument('--remove_displacements', default=False, help='Removing files in path data/displacements/ will save memory.')
    args = parser.parse_args()
    
    day_start, day_end = args.day_range
    
    with Pool(processes=1) as p:
        p.map(end_to_end, np.arange(day_start, day_end), args)
