'''
Created March 26, 2025

Script for generating motion simulation for a test case,
using publicly available data from the Calgary-Campinas dataset

'''

import os
os.environ['LD_LIBRARY_PATH'] ="/home/nghiemb/miniconda3/envs/GenPyMoCo_PTTF_BFGS_env/lib"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '' #TEMPORARILY MAKE GPU INVISIBLE

import numpy as np

import matplotlib.pyplot as plt
import pathlib as plib
from time import time

import torch
if torch.cuda.is_available():#check if torch is using gpu acceleration
    device = 'cuda'
else:
    device = 'cpu'

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
m_GT = torch.from_numpy(m_GT).to(device)
C = torch.from_numpy(C).to(device)

mask_bkg = rec.getMask(C)

Nx = m_GT.shape[0]
Ny = m_GT.shape[1]
Nz = m_GT.shape[2]
img_dims = [Nx,Ny,Nz]

#%%-----------------------------------------------------------------------------
#-------------------------------MOTION SIMULATION-------------------------------
#%%-----------------------------------------------------------------------------

#Generating discrete random motion trajectory
mild_specs = {'Tx':[0.2,0.1],'Ty':[0.4,0.2],'Tz':[0.4,0.2],\
            'Rx':[0.5,0.2],'Ry':[0.2,0.1],'Rz':[0.2,0.1]} #[max_rate, prob]
moderate_specs = {'Tx':[0.4,0.15],'Ty':[0.9,0.3],'Tz':[0.9,0.3],\
            'Rx':[1,0.3],'Ry':[0.5,0.15],'Rz':[0.5,0.15]} #[max_rate, prob]
severe_specs = {'Tx':[0.8,0.3],'Ty':[1.8,0.6],'Tz':[1.8,0.6],\
            'Rx':[2,0.6],'Ry':[1.0,0.3],'Rz':[1.0,0.3]} #2x the max_rate and probability
extreme_specs = {'Tx':[1.6,0.6],'Ty':[3.6,1.0],'Tz':[3.6,1.0],\
            'Rx':[4,1.0],'Ry':[2.0,0.6],'Rz':[2.0,0.6]} #4x the probability
motion_specs = {'mild':mild_specs,'moderate':moderate_specs,\
                'severe':severe_specs, 'extreme':extreme_specs}

#Generate sequential sampling pattern
pattern = "sequential" #interleaved or sequential
Rs = (1,1) #SENSE acceleration factor along PE1 and PE2
TR_shot = 16 #number of TRs per shot
U_array = np.transpose(msi.make_samp(np.transpose(m_GT.cpu(), (1,0,2)), \
                                     Rs, TR_shot, order=pattern), \
                                        (0,2,1,3)).astype(np.float32)

U = torch.from_numpy(U_array).to(device); del U_array

#Generate motion parameters
j = 1; k = 1 #legacy parameters, can have placeholder values of 1; TO DO --> REMOVE THESE PARAMS
rand_keys = msi._gen_key(sub, j, k)
motion_lv = 'moderate'
nshots = U.shape[0] #PE1 along Ny in this case

Mtraj_GT = torch.tensor(msi._gen_traj(rand_keys, motion_lv, nshots, motion_specs)).to(device)

T_GT = [Mtraj_GT[:,0], Mtraj_GT[:,1], Mtraj_GT[:,2]]
R_GT = [Mtraj_GT[:,3], Mtraj_GT[:,4], Mtraj_GT[:,5]]

T0 = [Mtraj_GT[:,0]*0, Mtraj_GT[:,1]*0, Mtraj_GT[:,2]*0]
R0 = [Mtraj_GT[:,3]*0, Mtraj_GT[:,4]*0, Mtraj_GT[:,5]*0]

plot_Mtraj(T_GT, R_GT, T0, R0, img_dims) #Plotting motion trajectories (groundtruth vs initial)


t1 = time()
s_corrupted = eop._E(m_GT,C,U,T_GT,R_GT,res) #if CPU: 68 seconds
t2 = time()
print("Elapsed time: {} sec".format(t2 - t1))

t3 = time()
CG_mask=None
CG_maxiter=5 #heuristically determined...
CG_nbatch=1 #batch for calls to E or EH
CG_result = "final"
CG_params = [CG_mask, CG_maxiter, CG_nbatch, CG_result]
m_corrupted, _, _ = rec.ImageEstimation(s_corrupted,C,U,T0,R0,res,CG_params) #if CPU: 
t4 = time()
print("Elapsed time: {} sec".format(t4 - t3))

plot_views(m_corrupted, vmax = 1.0)

# torch.save(Mtraj_GT, dpath + "/Mtraj_GT_seq_mild.pt")
# torch.save(s_corrupted, dpath + "/s_corrupted_seq_mild.pt")
# torch.save(m_corrupted, dpath + "/m_corrupted_seq_mild.pt")
