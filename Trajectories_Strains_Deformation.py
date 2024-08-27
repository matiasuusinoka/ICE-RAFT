import torch
import numpy as np
from glob import glob
from utils.trajectory_form import form_trajectories
from utils.straintensor_support import tensor_formulation, kinematic_matrix, DeformationGradient_quadrilateral, PolyArea
from multiprocessing import Pool

import os
import argparse

def displacements_to_strains(args):
    
    if 'all' in args.saved_quantities: saved_quantities = ['trajectories', 'strain_tensors', 'deformations']
    else: saved_quantities = args.saved_quantities
    
    scaling_factor = args.pixel_to_metric * args.resolution_ratio
    
    m = args.spatial_scale
    B = kinematic_matrix(m, scaling_factor)
    
    displacements = np.load(args.source_file) * scaling_factor
    trajectories = form_trajectories(displacements, args.pixel_to_metric, args.resolution_ratio)
    del displacements
    
    if args.use_velocity_filter:
        diff = np.diff(trajectories, axis=0)
        velo = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
        del diff
    
    if 'trajectories' in saved_quantities: 
        np.save(f'{args.save_path}/trajectories/{args.output_name}_trajectories', trajectories)
        
    # Remove the Eulerian displacements to save memory
    if args.remove_displacements == True: os.remove(args.source_file)
    
    if any(metric in saved_quantities for metric in ['strain_tensors','deformations']):

        #edge_lim = 60
        #trajectories = trajectories[:,:,edge_lim:-edge_lim,edge_lim:-edge_lim]
        
        # This is only for radar data with a stationary shadow sector
        resolution = trajectories.shape[2]
        
        if args.shadow_sector:
            trajectories[:,:,int(resolution/2):,int(resolution/2):] = np.nan

        for t in range(0, trajectories.shape[0]):

            # Calculating strain tensor so that we need displacement from original coordinates.
            # Strain rates are defined by the difference in strain tensor (defined later in "epsilon = np.diff(epsilonfull,axis=0)").
            u = (trajectories[t]-trajectories[0])
            F = np.array([[DeformationGradient_quadrilateral(u[:,x:x+m,y:y+m], B) for y in range(0,u.shape[1]-m,m-1)] for x in range(0,u.shape[2]-m,m-1)])
            del u

            epsilon = tensor_formulation(F, rotation_fixed = args.use_finite_strain)
            del F
            
            areas = np.array([[PolyArea(trajectories[t,:,x:x+m,y:y+m]) for y in range(0,trajectories.shape[2]-m,m-1)] 
                                                                        for x in range(0,trajectories.shape[3]-m,m-1)])

            if t == 0: epsilonfull, areasfull = (np.expand_dims(measure, axis=0) for measure in [epsilon, areas])
            else: epsilonfull, areasfull = (np.concatenate((measurefull,np.expand_dims(measure, axis=0))) for measurefull, measure in zip([epsilonfull, areasfull],[epsilon, areas]))
            del epsilon, areas

        del trajectories

        # Save the strain tensor
        if 'strain_tensors' in saved_quantities: 
            np.save(f'{args.save_path}/strain_tensors/{args.output_name}_straintensors',epsilonfull)


        if 'deformations' in saved_quantities:

            epsilon = np.diff(epsilonfull,axis=0)

            div = np.trace(epsilonfull, axis1=3, axis2=4)
            she = np.sqrt((epsilonfull[:,:,:,0,0] - epsilonfull[:,:,:,1,1])**2 + 4*epsilonfull[:,:,:,0,1]**2)
            tot = np.sqrt(div**2 + she**2)
            
            if args.use_velocity_filter:
                velofilter = np.array([[[velocity_filter(velo[t,x:x+m,y:y+m]) for y in range(0,velo.shape[1]-m,m-1)] 
                                                                                for x in range(0,velo.shape[2]-m,m-1)] 
                                                                               for t in range(1,velo.shape[0])])
                div,she,tot = (np.where(velofilter, measure, 0) for measure in [div,she,tot])
                del velofilter
            
            data = np.stack((div, she, tot, areasfull))
            del epsilon, div, she, tot, areasfull

            np.save(f'{args.save_path}/deformations/{args.output_name}_deformations', data)
           
           
if __name__ == '__main__':
               
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', default='demo_data/displacements/demo_displacements.npy', help="Give the path to and the name of (with the file extension) the desired file.")
    parser.add_argument('--save_path', default='demo_data/', help="Give the path to and the name of (with the file extension) the desired file.")
    parser.add_argument('--output_name', default='demo', help="Give the output files a prefix.")
    parser.add_argument('--saved_quantities', default=['trajectories', 'strain_tensors', 'deformations', 'all'], nargs='+', help="Define which quantities will be saved. Options: trajectories, strain_tensors and deformations", required=True)
    parser.add_argument('--spatial_scale', default = 2, type=int, help="Spatial scale defines the size of the deformations objects. 2 is pixel scale, 3 the scale of two pixels etc.")
    parser.add_argument('--pixel_to_metric', type=float, help="Define the spatial scale of one pixel i.e. how many meters does one pixel represent.", required=True)
    parser.add_argument('--resolution_ratio', type=float, help="Define the resolution ratio between the original images and the one used to produce the displacement fields.", required=True)
    parser.add_argument('--use_finite_strain', default = True, help='The deformations will either use infinitesimal or finite strain.')
    parser.add_argument('--use_velocity_filter', default = False, help='Remove deformation estimates with vertice velocity gradients below a given threshold. (Using the filter makes the program much slower).')
    parser.add_argument('--remove_displacements', default=False, help='Removing files in path data/displacements/ will save memory.')
    parser.add_argument('--shadow_sector', default=True, help='Removing a shadow sector from demo_data. This is generally not applicable to other radar datasets.')
    args = parser.parse_args()
           
    for path in [f'{args.save_path}/trajectories',f'{args.save_path}/strain_tensors',f'{args.save_path}/deformations']:
        if not os.path.isdir(path): os.mkdir(path)

    
    # Parallelization can be considered when processing multiple separate image sequences.
    #with Pool(processes=1) as p:
    #    p.map(displacements_to_strains, np.arange(day_start, day_end), args)
           
    displacements_to_strains(args)
