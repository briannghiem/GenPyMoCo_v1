'''
Functions for evaluating image quality metrics
'''

import numpy as np
import piq
import torch

#----------------------------------------------------------
def evalRMSE(m, m_gt):
    dif2 = (np.abs(m.flatten()) - np.abs(m_gt.flatten()))**2
    return np.sqrt(np.mean(dif2))

def evalPE(m, m_gt, mask=None): #percent error
    if np.all(mask==None):
        mask = torch.ones(m.shape, device='cuda')
    #
    m *= mask; m_gt *= mask
    return 100*(evalRMSE(m, m_gt) / evalRMSE(m_gt, np.zeros(m_gt.shape)))

def evalRMSE_ROI(m, m_gt, mask):
    dif2 = (np.abs(m.flatten()) - np.abs(m_gt.flatten()))**2
    dif2_ROI = dif2[np.where(mask.flatten() == 1)]
    return np.sqrt(np.mean(dif2_ROI))

def evalPE_ROI(m, m_gt, mask): #percent error
    RMSE_1 = evalRMSE_ROI(m, m_gt, mask)
    RMSE_2 = evalRMSE_ROI(m_gt, np.zeros(m_gt.shape), mask)
    return 100*(RMSE_1 / RMSE_2)

#----------------------------------------------------------
def evalSSIM(m, m_gt, mask=None): #percent error
    max_val = abs(m).flatten().max()
    if np.all(mask==None):
        mask = torch.ones(m.shape, device='cuda')
    #
    m *= mask; m_gt *= mask
    return np.mean(piq.ssim(abs(m_gt), abs(m), data_range = max_val).numpy())


#----------------------------------------------------------
def entropy(m, mask=None): #Pixel entropy, defined by Atkinson et al MRM 1999
    if np.all(mask==None):
        mask = torch.ones(m.shape, device='cuda')
    #
    m *= mask
    eps = 1e-15
    m+= eps #alternative to removing zeros, which would skew distribution
    B = abs(m)
    B_max = torch.sqrt(torch.sum(abs(m)**2))
    v = B/B_max
    H = -torch.sum(v * torch.log(v))
    return H

def GradientEntropy(m, mask=None):
    if np.all(mask==None):
        mask = torch.ones(m.shape, device='cuda')
    #
    Dx = m[1:,:,:] - m[:-1,:,:]
    Dy = m[:,1:,:] - m[:,:-1,:]
    Dz = m[:,:,1:] - m[:,:,:-1]
    Hx = entropy(Dx, mask[1:,:,:])
    Hy = entropy(Dy, mask[:,1:,:])
    Hz = entropy(Dz, mask[:,:,1:])
    return Hx, Hy, Hz

#----------------------------------------------------------
def bounding_box(mask, tol = 1e-2):
    x_proj = np.where(np.sum(mask, axis = (1,2))<tol,0,1) #binarized projection
    y_proj = np.where(np.sum(mask, axis = (0,2))<tol,0,1)
    z_proj = np.where(np.sum(mask, axis = (0,1))<tol,0,1)
    #
    x_inds = np.where(x_proj == 1)[0]
    y_inds = np.where(y_proj == 1)[0]
    z_inds = np.where(z_proj == 1)[0]
    #
    x_min = x_inds[0]; x_max = x_inds[-1]
    y_min = y_inds[0]; y_max = y_inds[-1]
    z_min = z_inds[0]; z_max = z_inds[-1]
    return x_min, x_max, y_min, y_max, z_min, z_max


def evalSSIM_bbox(m, m_gt, mask, tol = 1e-2):
    bbox = bounding_box(mask, tol)
    m_bbox = m[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
    m_GT_bbox = m_gt[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
    return evalSSIM(m_bbox, m_GT_bbox)

#----------------------------------------------------------
def float2int8(img):
    '''Source: https://stackoverflow.com/questions/53235638/how-should-i-convert-a-float32-image-to-an-uint8-image'''
    img = abs(img).detach()
    vmin = img.min()
    vmax = img.max() - vmin
    img_int8 = ((img - vmin)/vmax) * (2**8 - 1)
    return torch.tensor(img_int8, dtype=torch.uint8, device='cuda')

