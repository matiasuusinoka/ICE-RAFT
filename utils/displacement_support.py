import torchvision.transforms.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple, Union
import torch

class OpticalFlow(nn.Module):
    def forward(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(img1, Tensor): img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor): img2 = F.pil_to_tensor(img2)

        img1, img2 = F.convert_image_dtype(img1, torch.float), F.convert_image_dtype(img2, torch.float)
        img1, img2 = F.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous(), F.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous()
        return img1, img2

def preprocess(img1_batch, img2_batch, resolution):
    transforms = OpticalFlow()
    img1_batch = F.resize(img1_batch, size=[resolution, resolution], antialias=False)
    img2_batch = F.resize(img2_batch, size=[resolution, resolution], antialias=False)
    return transforms(img1_batch, img2_batch)


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1, y1 = x0 + dx,  y0 + dy
    x1,y1,dx,dy = (variab.reshape(-1) for variab in [x1,y1,dx,dy])
    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1,y1,dx,dy = (variab[valid] for variab in [x1,y1,dx,dy])
    flow_x, flow_y = (interpolate.griddata((x1, y1), variab, (x0, y0), method='nearest', fill_value=0) for variab in [dx, dy])
    
    return torch.from_numpy(np.stack([flow_x, flow_y], axis=0)).float()


def bf_stepping(step, conscale):
    if step < 3: backstep,frontstep = 0, conscale
    elif step > len(filenames)-(conscale-1): backstep,frontstep = len(filenames) - (conscale-1), len(filenames) - 1
    else: backstep, frontstep = step - int(conscale/2), step + int(conscale/2)
    return backstep, frontstep
