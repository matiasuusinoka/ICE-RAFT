import numpy as np
import argparse
import os

def gt_rigidbody_rotated(t=0, scale='small'):
    
    degmax = 5 if scale == 'small' else 30
    deg = np.linspace(0,degmax,100)[t]

    theta = np.radians(-deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    x, y = np.meshgrid(np.linspace(0,10000, 1000), np.linspace(0,10000,1000))

    for xs in range(0,1000):
        for ys in range(0,1000): x[xs,ys], y[xs,ys] = np.dot(R, (np.array([x[xs,ys], y[xs,ys]])-5000)).T+5000

    return np.stack((x,y))

def gt_uniform_shear(t=0, scale='small'):
    
    shear_max = -.09 if scale == 'small' else -.3

    gy = np.linspace(0, shear_max, 100)[t]
    ex, ey, gx = (0 for _ in range(3))
    P0 = np.array(np.meshgrid(np.linspace(-5000, 5000, 1000), np.linspace(-5000, 5000, 1000)))

    return np.array([(1+ex)*P0[0] + gx*P0[1], (1+ey)*P0[1]+gy*P0[0]])+5000

def gt_uniform_shear_rotated(t=0, scale='small'):
    
    shear_max = -.09 if scale == 'small' else -.3
    degmax = 5 if scale == 'small' else 30
    deg = np.linspace(0,degmax,100)[t]

    theta = np.radians(-deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    gy = np.linspace(0, shear_max, 100)[t]
    ex, ey, gx = (0 for _ in range(3))
    P0 = np.array(np.meshgrid(np.linspace(-5000, 5000, 1000), np.linspace(-5000, 5000, 1000)))
    
    x,y = np.array([(1+ex)*P0[0] + gx*P0[1], (1+ey)*P0[1]+gy*P0[0]])+5000

    for xs in range(0,1000):
        for ys in range(0,1000): x[xs,ys], y[xs,ys] = np.dot(R, (np.array([x[xs,ys], y[xs,ys]])-5000)).T+5000

    return np.stack((x,y))

def gt_localized_shear(t=0, scale='small'):
    
    if scale == 'large': t = t*5
    
    x, y = np.meshgrid(np.linspace(0,10000, 1000), np.linspace(0,10000,1000))
    x[500:, :] += (t*10/1.2)
    x[:500, :] -= (t*10/1.2)
    
    return np.stack((x,y))

def gt_localized_shear_rotated(t=0, scale='small'):
    
    degmax = 5 if scale == 'small' else 30
    deg = np.linspace(0,degmax,100)[t]
    if scale == 'large': t = t*5
        
    theta = np.radians(-deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    x, y = np.meshgrid(np.linspace(0,10000, 1000), np.linspace(0,10000,1000))
    x[500:, :] += (t*10/1.2)
    x[:500, :] -= (t*10/1.2)
    
    for xs in range(0,1000):
        for ys in range(0,1000): x[xs,ys], y[xs,ys] = np.dot(R, (np.array([x[xs,ys], y[xs,ys]])-5000)).T+5000
    
    return np.stack((x,y))


def generate_groundtruth(args):
    
    from tqdm import tqdm
    
    # Choosing the method
    if args.motion_field == 'rigidbody_rotated': method = gt_rigidbody_rotated
    elif args.motion_field == 'uniform_shear': method = gt_uniform_shear
    elif args.motion_field == 'uniform_shear_rotated': method = gt_uniform_shear_rotated
    elif args.motion_field == 'localized_shear': method = gt_localized_shear
    elif args.motion_field == 'localized_shear_rotated': method = gt_localized_shear_rotated
    else: print("\nMisspelled motion field:\nThe motion_field has to be one of the following: 'rigidbody_rotated', 'uniform_shear', 'uniform_shear_rotated', 'localized_shear', 'localized_shear_rotated'\n"); return
        
    print(f'Generating {args.motion_field} ground truth with {args.scale} displacements.')
    
    for t in tqdm(range(args.num_tsteps)):

        gt = method(t, scale=args.scale)

        gt = np.expand_dims(gt, axis=0)
        if t == 0: gt_trajectories = gt.copy()
        gt_trajectories = np.concatenate((gt_trajectories, gt))

    np.save(f'{args.save_path}/{args.scale}/{args.motion_field}', gt_trajectories)


if __name__ == '__main__':
               
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_field', help="Choose which motion field to produce. The option are 'rigidbody_rotated', 'uniform_shear', 'uniform_shear_rotated', 'localized_shear', 'localized_shear_rotated'.", required=True)
    parser.add_argument('--save_path', default='ground_truth', help="Give the folder for the save file.")
    parser.add_argument('--scale', help="Choose the scale of the displacements from either 'small' or 'large.", required=True)
    parser.add_argument('--num_tsteps', default=10, help="Choose the number of timesteps considered in the verification.")
    args = parser.parse_args()
           
    if not os.path.isdir(f'./{args.save_path}/{args.scale}/'):
        os.makedirs(f'./{args.save_path}/{args.scale}/')

    generate_groundtruth(args)
