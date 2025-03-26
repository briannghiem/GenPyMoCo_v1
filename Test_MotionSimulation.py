'''
Created December 10, 2024

Profiling loss functions for "multi-modal" image registration
Of k-space segments (shots)

Goal is to assess the loss landscape --> is my idea even viable?

This is a script for sliding window recons of segments of k-space
'''

import os
# os.environ['LD_LIBRARY_PATH'] ="/home/nghiemb/miniconda3/envs/GenPyMoCo_PTTF_env/lib"
os.environ['LD_LIBRARY_PATH'] ="/home/nghiemb/miniconda3/envs/GenPyMoCo_PTTF_BFGS_env/lib"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

import matplotlib.pyplot as plt
import pathlib as plib

import torch
# torch.cuda.is_available() #check if torch is using gpu acceleration

import encode.encode_op as eop
import recon.recon_op as rec
import utils.metrics as mtc
import motion.motion_sim as msi
from utils.visualize import plot_views, plot_Mtraj


#%%-----------------------------------------------------------------------------
#-----------------------------------LOAD DATA-----------------------------------
#%%-----------------------------------------------------------------------------

#Set paths
root = r'/home/nghiemb/GenPyMoCo/data/archived'
paradigm = 'C' #C, D, E, F
sub = 1 #1, 4, 5, 6, 7

sub_path = root + r'/Sub{}'.format(sub) #path to img_CG.npy and sens.npy
test_path = root + r'/Paradigm_1{}/Test{}'.format(paradigm, sub) #path to Mtraj, m_corrupted, and s_corrupted

#Import data
m_GT = np.load(sub_path + r'/img_CG.npy') #loaded with correct orientation (LR, AP, SI) and padding, per old simulation scripts
C = np.load(sub_path + r'/sens.npy')
res = np.array([1,1,1]) #image resolution: 1 mm iso

#Convert to torch tensor
m_GT = torch.from_numpy(m_GT).to('cuda')
C = torch.from_numpy(C).to('cuda')


mask_bkg = rec.getMask(C)

Nx = m_GT.shape[0]
Ny = m_GT.shape[1]
Nz = m_GT.shape[2]
img_dims = [Nx,Ny,Nz]

#%%-----------------------------------------------------------------------------
#-------------------------------MOTION SIMULATION-------------------------------
#%%-----------------------------------------------------------------------------

U = np.load(root + r'/U.npy')
Mtraj_GT = np.load(test_path + r'/Mtraj.npy')
nshots = U.shape[0]

s_corrupted = np.load(test_path + r'/s_corrupted.npy')
m_corrupted = np.load(test_path + r'/m_corrupted.npy')


U = torch.from_numpy(U).to('cuda')
Mtraj_GT = torch.from_numpy(Mtraj_GT).to('cuda')

s_corrupted = torch.from_numpy(s_corrupted).to('cuda')
m_corrupted = torch.from_numpy(m_corrupted).to('cuda')

# plot_views(abs(m_GT).detach().cpu(), 'auto')
# plot_views(abs(m_corrupted).detach().cpu(), 'auto')


T_GT = [Mtraj_GT[:,0], Mtraj_GT[:,1], Mtraj_GT[:,2]]
R_GT = [Mtraj_GT[:,3], Mtraj_GT[:,4], Mtraj_GT[:,5]]
# plot_Mtraj(T_GT, R_GT, T_GT, R_GT, img_dims, rescale = 0)


#%%-----------------------------------------------------------------------------
#---------------------------------PROFILING UNet--------------------------------
#%%-----------------------------------------------------------------------------

wpath_SAPUNet = r'/home/nghiemb/GenPyMoCo/cnn/weights/_archived'
pads = [11,3]

UNet_params = [wpath_SAPUNet, pads]
m_est = m_corrupted
from time import time
t1 = time()
output_UNet = rec.UNet_magnitude(m_est, UNet_params)
t2 = time()
print("Elapsed time: {} sec".format(t2 - t1))
# m_out_UNet = output_UNet[0]
# plot_views(abs(m_out_UNet).detach().cpu(), 'auto')

#%%-----------------------------------------------------------------------------
#----------------------------------PROFILING JE---------------------------------
#%%-----------------------------------------------------------------------------

offset = 0
Tx_init = torch.zeros(nshots)+offset; Tx_init[0] = 0
Ty_init = torch.zeros(nshots)+offset; Ty_init[0] = 0
Tz_init = torch.zeros(nshots)+offset; Tz_init[0] = 0
Rx_init = torch.zeros(nshots)+offset; Rx_init[0] = 0
Ry_init = torch.zeros(nshots)+offset; Ry_init[0] = 0
Rz_init = torch.zeros(nshots)+offset; Rz_init[0] = 0
T_init = [Tx_init.to('cuda'), Ty_init.to('cuda'), Tz_init.to('cuda')]
R_init = [Rx_init.to('cuda'), Ry_init.to('cuda'), Rz_init.to('cuda')]

T = T_init
R = R_init

JE_maxiters = 10
ME_params = ['Adam', 'DC', 1, 1, 0.05]
CG_params = [None, 5, 1, 'optimal']
g_window = 10
JE_params = [ME_params, CG_params, JE_maxiters, g_window]
m_est = m_corrupted
output_JE = rec.JE(m_est, s_corrupted, C, \
                   U, T, R, res, JE_params)

m_out_JE = output_JE[0]
T_JE = output_JE[1]
R_JE = output_JE[2]

plot_views(abs(m_out_JE).detach().cpu(), 'auto')
plot_Mtraj(T_GT, R_GT, T_JE, R_JE, img_dims, rescale = 0)


DC_list_JE = [val.detach().cpu().numpy() for val in output_JE[-1]]
DC_array_JE = np.array(DC_list_JE)
iter_array = np.arange(0, JE_maxiters+1)

plt.figure()
plt.plot(iter_array, DC_array_JE)
plt.xlabel("Iteration")
plt.ylabel("Total DC loss")
plt.title("Total DC Loss Across {} JE Iterations".format(JE_maxiters))
plt.show()


'''
from torchmin import minimize
from torchmin.benchmarks import rosen

def _f(x):
    return sum(x.flatten())**2 +1

# initial point
x0 = torch.zeros(2,2).to("cuda")

# BFGS
result = minimize(_f, x0, method='bfgs')

# Newton Conjugate Gradient
result = minimize(rosen, x0, method='newton-cg')
'''

#%%-----------------------------------------------------------------------------
#-------------------------------PROFILING UNet+JE-------------------------------
#%%-----------------------------------------------------------------------------

offset = 0
Tx_init = torch.zeros(nshots)+offset; Tx_init[0] = 0
Ty_init = torch.zeros(nshots)+offset; Ty_init[0] = 0
Tz_init = torch.zeros(nshots)+offset; Tz_init[0] = 0
Rx_init = torch.zeros(nshots)+offset; Rx_init[0] = 0
Ry_init = torch.zeros(nshots)+offset; Ry_init[0] = 0
Rz_init = torch.zeros(nshots)+offset; Rz_init[0] = 0
T_init = [Tx_init.to('cuda'), Ty_init.to('cuda'), Tz_init.to('cuda')]
R_init = [Rx_init.to('cuda'), Ry_init.to('cuda'), Rz_init.to('cuda')]

T = T_init
R = R_init

wpath_SAPUNet = r'/home/nghiemb/GenPyMoCo/cnn/weights/_archived'
pads = [11,3]

UNetJE_maxiters = 10
ME_params = ['Adam', 'DC', 1, 1, 0.05]
CG_params = [None, 5, 1, 'optimal']
g_window = 10
JE_params = [ME_params, CG_params, JE_maxiters, g_window]
UNet_params = [wpath_SAPUNet, pads]
m_est = m_corrupted
output_UNetJE = rec.UNetJE(m_est, s_corrupted, C, \
                           U, T, R, res, JE_params, UNet_params)

m_out_UNetJE = output_UNetJE[0]
T_UNetJE = output_UNetJE[1]
R_UNetJE = output_UNetJE[2]

plot_views(abs(m_out_UNetJE).detach().cpu(), 'auto')
plot_Mtraj(T_GT, R_GT, T_UNetJE, R_UNetJE, img_dims, rescale = 0)

DC_list_UNetJE = [val.detach().cpu().numpy() for val in output_UNetJE[-1]]
DC_array_UNetJE = np.array(DC_list_UNetJE)
iter_array = np.arange(0, UNetJE_maxiters+1)

plt.figure()
plt.plot(iter_array, DC_array_UNetJE)
plt.xlabel("Iteration")
plt.ylabel("Total DC loss")
plt.title("Total DC Loss Across {} UNet+JE Iterations".format(UNetJE_maxiters))
plt.show()
