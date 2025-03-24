import os
import numpy as np
os.environ['DDE_BACKEND'] = "pytorch" #set backend for DeepXDE

import torch
import deepxde as xde

import encode.encode_op as eop
import recon.recon_op as rec
import utils.metrics as mtc
import motion.motion_sim as msi

#---------------------------------------------------------------------------
#Set paths
root = r'/home/nghiemb/GenPyMoCo'
case = 1
dpath = root + r'/data/synthetic/Sub{}'.format(case)

#Import data
m_GT_init = np.load(dpath + r'/img_GT.npy') #initial orientation SI, LR, AP
m_GT = np.pad(m_GT_init[:,:,:,0,0] + 1j*m_GT_init[:,:,:,1,0], ((1,1), (0,0), (0,0)))
del m_GT_init
C = np.load(dpath + r'/sens.npy')
res = np.array([1,1,1]) #image resolution: 1 mm iso
#Transpose to reorient as LR, AP, SI
m_GT = np.transpose(m_GT, (1,2,0))
m_GT = np.abs(m_GT[6:-6, 3:-3, :])

#Convert to torch tensor
m_GT = torch.from_numpy(m_GT).to('cuda')
C = torch.from_numpy(C).to('cuda')

Nx = m_GT.shape[0]
Ny = m_GT.shape[1]
Nz = m_GT.shape[2]

# mask = rec.getMask(C) #mask of coverage of coil profiles from BART estimation
# cerebrum_mask = np.ones(m_GT.shape) #Set mask to identity for simulations; otherwise identify base of cerebellum

#---------------------------------------------------------------------------
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


j = 1; k = 1 #legacy parameters, from training dataset script
rand_keys = msi._gen_key(60+case, j, k)
motion_lv = 'mild'
nshots = Ny #PE1 along Ny in this case
# nshots = 16 #PE1 along Ny in this case

U = torch.zeros(torch.Size((nshots,1))+m_GT.shape, dtype=m_GT.dtype)
for i in range(nshots):
    U[i,:,:,i,:] = 1

Mtraj_GT = torch.tensor(msi._gen_traj(rand_keys, motion_lv, nshots, motion_specs))

T = [Mtraj_GT[:,0]/(Nx/2), Mtraj_GT[:,1]/(Nx/2), Mtraj_GT[:,2]/(Nx/2)]
R = [Mtraj_GT[:,3]*(np.pi/180), Mtraj_GT[:,4]*(np.pi/180), Mtraj_GT[:,5]*(np.pi/180)]

T0 = [Mtraj_GT[:,0]*0, Mtraj_GT[:,1]*0, Mtraj_GT[:,2]*0]
R0 = [Mtraj_GT[:,3]*0, Mtraj_GT[:,4]*0, Mtraj_GT[:,5]*0]

from time import time
t1 = time()
s_corrupted = eop._E(m_GT,C,U,T,R,batch=1) #11.9 sec for sequential eval of 218 motion states
t2 = time()
print("Elapsed time: {} sec".format(t2 - t1))

t3 = time()
m_corrupted = eop._EH(s_corrupted,C,U,T0,R0,batch=1) #13.9 sec for sequential eval of 218 motion states
t4 = time()
print("Elapsed time: {} sec".format(t4 - t3))

#---------------------------------------------------------------------------

#DC only for central half of k-space
nshots = Ny//2 #PE1 along Ny in this case
s_corrupted_lowres = s_corrupted[:,Nx//4:-Nx//4, Ny//4:-Ny//4, Nz//4:-Nz//4]
m_corrupted_lowres = torch.sqrt((eop.IFFT(s_corrupted_lowres, dims=(1,2,3))**2).sum(axis=0))

C_FT = eop.FFT(C, dims = (1,2,3))
C_lowres = eop.IFFT(C_FT[:, Nx//4:-Nx//4, Ny//4:-Ny//4, Nz//4:-Nz//4], dims = (1,2,3))
U_lowres = U[Ny//4:Ny//4+nshots,:,Nx//4:-Nx//4, Ny//4:-Ny//4, Nz//4:-Nz//4]

Tx = torch.nn.Parameter(data=torch.zeros(nshots), requires_grad=True)
Ty = torch.nn.Parameter(data=torch.zeros(nshots), requires_grad=True)
Tz = torch.nn.Parameter(data=torch.zeros(nshots), requires_grad=True)
Rx = torch.nn.Parameter(data=torch.zeros(nshots), requires_grad=True)
Ry = torch.nn.Parameter(data=torch.zeros(nshots), requires_grad=True)
Rz = torch.nn.Parameter(data=torch.zeros(nshots), requires_grad=True)
T = [Tx, Ty, Tz]
R = [Rx, Ry, Rz]

nbatch = 16
maxiter = 1
updates = rec.MotionEstimation(m_corrupted_lowres,C_lowres,U_lowres,\
                               s_corrupted_lowres,T,R,batch=nbatch,maxiter=maxiter)

#%%-----------------------------------------------------------------------------
#--------------------------------IMAGE ESTIMATION-------------------------------
#%%-----------------------------------------------------------------------------
CG_mask=None
CG_maxiter=3
CG_nbatch=1 #batch for calls to E or EH
CG_params = [CG_mask, CG_maxiter, CG_nbatch]

t1 = time()
m_est = rec.ImageEstimation(s_corrupted,C,U,T,R,CG_params)
t2 = time()
print("Elapsed time: {} sec".format(t2 - t1))

# #---------------------------------------------------------------------------
# from utils.visualize import plot_views

# plot_views(abs(m_corrupted.detach().cpu()))
# plot_views(abs(m_est.detach().cpu()))

# import matplotlib.pyplot as plt
# Mtraj_temp = Mtraj_GT.detach().cpu()

# plt.figure()
# plt.plot(Mtraj_temp[:,0], label="Tx")
# plt.plot(Mtraj_temp[:,1], label="Ty")
# plt.plot(Mtraj_temp[:,2], label="Tz")
# plt.legend(loc="upper left")
# plt.ylabel("Translations (mm)")
# plt.show()

# plt.figure()
# plt.plot(Mtraj_temp[:,3], label="Rx")
# plt.plot(Mtraj_temp[:,4], label="Ry")
# plt.plot(Mtraj_temp[:,5], label="Rz")
# plt.legend(loc="upper left")
# plt.ylabel("Rotations (deg)")
# plt.show()

