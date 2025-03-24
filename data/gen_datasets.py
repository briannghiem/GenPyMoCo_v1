"""
Created November 8, 2024

Script for generating training and testing datasets
"""

import os
import pathlib as plib
import shutil

import numpy as np
os.environ['DDE_BACKEND'] = "pytorch" #set backend for DeepXDE

import torch
import deepxde as xde

import encode.encode_op as eop
import recon.recon_op as rec
import utils.metrics as mtc
import motion.motion_sim as msi
from utils.visualize import plot_views, plot_Mtraj


#Set paths
root = r'/home/nghiemb/GenPyMoCo'
dpath = r'/home/nghiemb/Data/CC'
spath_train = os.path.join(root,'data','train')
spath_test = os.path.join(root,'data','test')
spath_val = os.path.join(root,'data','val')

#List of files names from CC dataset directory
m_files = sorted(os.listdir(os.path.join(dpath,'m_complex'))) #alphanumeric order
C_files = sorted(os.listdir(os.path.join(dpath,'sens')))


#Generating discrete random motion trajectory
mild_specs = {'Tx':[0.2,0.1/14],'Ty':[0.4,0.2/14],'Tz':[0.4,0.2/14],\
            'Rx':[0.5,0.2/14],'Ry':[0.2,0.1/14],'Rz':[0.2,0.1/14]} #[max_rate, prob]
moderate_specs = {'Tx':[0.4,0.15/14],'Ty':[0.9,0.3/14],'Tz':[0.9,0.3/14],\
            'Rx':[1,0.3/14],'Ry':[0.5,0.15/14],'Rz':[0.5,0.15/14]} #[max_rate, prob]
large_specs = {'Tx':[0.8,0.3/14],'Ty':[1.8,0.6/14],'Tz':[1.8,0.6/14],\
            'Rx':[2,0.6/14],'Ry':[1.0,0.3/14],'Rz':[1.0,0.3/14]} #2x the max_rate and probability
extreme_specs = {'Tx':[1.6,0.6/14],'Ty':[3.6,1.0/14],'Tz':[3.6,1.0/14],\
            'Rx':[4,1.0/14],'Ry':[2.0,0.6/14],'Rz':[2.0,0.6/14]} #4x the probability
motion_specs = {'mild':mild_specs,'moderate':moderate_specs,\
                'large':large_specs, 'extreme':extreme_specs}

# motion_lv_list = ['mild', 'moderate', 'large', 'extreme']
motion_lv_list = ['moderate', 'large']
# nsims = 1 #number of simulations per motion level in above list
nsims_list = [0,1] #number of simulations per motion level in above list

# Sub48 has coil profile estimation issues due to large wrap-around artifacts, 
# and Sub55 has noticeable motion artifacts
exclude_list = [48, 55] 


for i in range(len(m_files)):
   print("Subject {}".format(i+1))
   if i+1 in exclude_list:
      pass
   else:
      #%%-----------------------------------------------------------------------------
      #----------------------------------LOADING DATA---------------------------------
      #%%-----------------------------------------------------------------------------
      # Load data
      m_GT = np.load(os.path.join(dpath,'m_complex',m_files[i])) #oriented as LR, AP, SI
      C = np.load(os.path.join(dpath,'sens',C_files[i])) #initial orientation: AP, SI, LR, NC
      C = np.transpose(C, (3,2,0,1)) #reorient as NC, LR, AP, SI
      #
      spath_init = os.path.join(spath_train,'Sub{}'.format(i+1))
      plib.Path(spath_init).mkdir(parents=True, exist_ok=True)
      #
      #-------------------------------
      # Convert to torch tensor
      m_GT = torch.from_numpy(m_GT).to('cuda')
      C = torch.from_numpy(C).to('cuda')
      res = np.array([1,1,1]) #image resolution: 1 mm iso
      #
      #-------------------------------
      # Zero pad to image dimensions RO x PE1 x PE2 = 256 x 256 x 192
      dim_current = torch.tensor(m_GT.shape)
      dim_target = torch.tensor([192,256,256]) #PE2, PE1, RO
      pad_vals = dim_target - dim_current
      #
      m_GT = torch.nn.functional.pad(m_GT, (pad_vals[-1]//2,pad_vals[-1]//2,\
                                                pad_vals[-2]//2,pad_vals[-2]//2,\
                                                pad_vals[-3]//2,pad_vals[-3]//2))
      C = torch.nn.functional.pad(C, (pad_vals[-1]//2,pad_vals[-1]//2,\
                                                pad_vals[-2]//2,pad_vals[-2]//2,\
                                                pad_vals[-3]//2,pad_vals[-3]//2,0,0))
      #
      mask_bkg = rec.getMask(C)
      Nx = m_GT.shape[0]
      Ny = m_GT.shape[1]
      Nz = m_GT.shape[2]
      img_dims = [Nx,Ny,Nz]
      #
      #Generate sequential sampling pattern
      pattern = "sequential"
      Rs = (2,1) #PE1, PE2
      TR_shot = 1 #number of TRs per shot
      U_array = np.transpose(msi.make_samp(np.transpose(m_GT.cpu(), (1,0,2)), \
                                          Rs, TR_shot, order=pattern), \
                                             (0,2,1,3)).astype(np.float32)
      U = torch.from_numpy(U_array).to('cuda'); del U_array
      #
      torch.save(U, spath_init + "/U.pt")
      torch.save(C, spath_init + "/C.pt")
      torch.save(m_GT, spath_init + "/m_GT.pt")
      #
      #%%-----------------------------------------------------------------------------
      #-------------------------------MOTION SIMULATION-------------------------------
      #%%-----------------------------------------------------------------------------
      #Generate motion parameters
      for j, motion_lv in enumerate(motion_lv_list):
         for k in nsims_list:
            spath_temp = os.path.join(spath_init,motion_lv,'sim{}'.format(k+1))
            plib.Path(spath_temp).mkdir(parents=True, exist_ok=True)
            rand_keys = msi._gen_key(i, j, k)
            nshots = U.shape[0] #PE1 along Ny in this case
            #
            Mtraj_GT = torch.tensor(msi._gen_traj(rand_keys, motion_lv, nshots, motion_specs))
            #
            T_GT = [Mtraj_GT[:,0], Mtraj_GT[:,1], Mtraj_GT[:,2]]
            R_GT = [Mtraj_GT[:,3], Mtraj_GT[:,4], Mtraj_GT[:,5]]
            #
            T0 = [Mtraj_GT[:,0]*0, Mtraj_GT[:,1]*0, Mtraj_GT[:,2]*0]
            R0 = [Mtraj_GT[:,3]*0, Mtraj_GT[:,4]*0, Mtraj_GT[:,5]*0]
            #
            torch.save(Mtraj_GT, spath_temp + "/Mtraj_GT.pt")
            # plot_Mtraj(T_GT, R_GT, T_GT, R_GT, img_dims, rescale = 0)
            #
            from time import time
            t1 = time()
            s_corrupted = eop._E(m_GT,C,U,T_GT,R_GT,res) #11.9 sec for sequential eval of 218 motion states
            t2 = time()
            print("Elapsed time: {} sec".format(t2 - t1))
            torch.save(s_corrupted, spath_temp + "/s_corrupted.pt")
            #
            t3 = time()
            CG_mask=None
            CG_maxiter=5 #heuristic
            CG_nbatch=1 #batch for calls to E or EH
            CG_result = "optimal"
            CG_params = [CG_mask, CG_maxiter, CG_nbatch, CG_result]
            m_corrupted, _, _ = rec.ImageEstimation(s_corrupted,C,U,T0,R0,res,CG_params)
            t4 = time()
            print("Elapsed time: {} sec".format(t4 - t3))
            torch.save(m_corrupted, spath_temp + "/m_corrupted.pt")


#%%-----------------------------------------------------------------------------
#-----------------------------------POST-HOC------------------------------------
#-----------------------SPLITTING DATA FOR TRAIN/VAL/TEST-----------------------
#%%-----------------------------------------------------------------------------
def del_ind(a, b):
   '''
   Find index of elements (b) to be deleted from array (a)
   '''
   return np.array([np.where(a == val)[0][0] for val in b])

exclude_array = np.asarray(exclude_list)-1
ID_total = np.arange(1, 68)
ID_sub = np.delete(ID_total, exclude_array)

#------------------------------------
#Define train / val / test split 
N_test = 5
N_train = len(ID_sub) - N_test
N_val = int(0.2*N_train)

#Random choice generator
rand_key = 0
rng = np.random.default_rng(seed=rand_key)

#Selecting participants for testing dataset
ID_test = np.sort(rng.choice(a=ID_sub, size=(N_test,), replace=0)) #binary array

#Selecting participants for validation
ID_train_init = np.delete(ID_sub, del_ind(ID_sub, ID_test))
ID_val = np.sort(rng.choice(a=ID_train_init, size=(N_val,), replace=0)) #binary array

#Selecting participants for training dataset
ID_train = np.delete(ID_train_init, del_ind(ID_train_init, ID_val))

#------------------------------------
# Moving files to /test and /val
for ind in ID_test:
   try:
      temp_path = os.path.join(spath_train,'Sub{}'.format(ind))
      shutil.move(temp_path, spath_test)
   except:
      print("Folder does not exist")

for ind in ID_val:
   try:
      temp_path = os.path.join(spath_train,'Sub{}'.format(ind))
      shutil.move(temp_path, spath_val)
   except:
      print("Folder does not exist")
