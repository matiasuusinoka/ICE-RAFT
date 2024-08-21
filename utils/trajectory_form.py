import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def bilinear_interpolation(x_in, y_in, f_in, x_out, y_out):
    f_out = np.zeros((y_out.size, x_out.size))
    for i in prange(f_out.shape[1]):
        idx = np.searchsorted(x_in, x_out[i])
        x1, x2, x = x_in[idx-1], x_in[idx], x_out[i]
        for j in prange(f_out.shape[0]):
            idy = np.searchsorted(y_in, y_out[j])
            y1, y2, y = y_in[idy-1], y_in[idy], y_out[j]
            f11, f21 = f_in[idy-1, idx-1], f_in[idy-1, idx]
            f12, f22 = f_in[idy, idx-1], f_in[idy, idx]
            f_out[j, i] = ((f11*(x2-x)*(y2-y)+f21*(x-x1)*(y2-y)+f12*(x2-x)*(y-y1)+f22*(x-x1)*(y-y1))/((x2-x1)*(y2-y1)))
    return f_out

def form_trajectories(displacement_map, pixel_to_metric, resolution_ratio):
    
    # We use the scaling factor to relate the trajectories back to the scale of the original radar data
    scaling_factor = pixel_to_metric * resolution_ratio
    
    x, y = np.arange(0,displacement_map.shape[2])*scaling_factor, np.arange(0,displacement_map.shape[3])*scaling_factor
    
    interpolation_factor = 10
    x2, y2 = np.linspace(0, displacement_map.shape[2], displacement_map.shape[2]*interpolation_factor)*scaling_factor, np.linspace(0, displacement_map.shape[3], displacement_map.shape[3]*interpolation_factor)*scaling_factor
    
    X, Y = np.meshgrid(x, y)

    coords_x, coords_y = ((X + displacement_map[0,0])), ((Y + displacement_map[0,1]))
    trajectories = np.expand_dims(np.stack((coords_x, coords_y)),axis=0)

    for t in range(1,displacement_map.shape[0]):
        disp_x, disp_y = bilinear_interpolation(x, y, displacement_map[t,0], x2, y2), bilinear_interpolation(x, y, displacement_map[t,1], x2, y2)

        for xes in range(coords_x.shape[0]):
            for yes in range(coords_x.shape[1]):
                try:
                    coords_x[xes,yes] += disp_x[int(coords_x[xes,yes] / resolution_ratio),int(coords_y[xes,yes] / resolution_ratio)]
                    coords_y[xes,yes] += disp_y[int(coords_x[xes,yes] / resolution_ratio),int(coords_y[xes,yes] / resolution_ratio)]
                except: continue

        del disp_x, disp_y
        trajectories = np.concatenate((trajectories,np.expand_dims(np.stack((coords_x, coords_y)),axis=0)),axis=0)

    return trajectories
