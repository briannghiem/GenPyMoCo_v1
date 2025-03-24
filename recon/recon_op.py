"""
Defining motion correction methods: UNet, Joint Estimation, and UNet-Assisted Joint Estimation
For UNet-Assisted Joint Motion and Image Estimation
"""

import os
os.environ['LD_LIBRARY_PATH'] ="/home/nghiemb/miniconda3/envs/GenPyMoCo_PTTF_env/lib"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true" #turn off GPU pre-allocation for TF

import numpy as np
import torch
import tensorflow as tf

from time import time
from functools import partial

import encode.encode_op as eop
import cnn.run_unet as cnn
import utils.metrics as mtc

from scipy.ndimage import rotate

from functools import partial
import torchmin #scipy-like minimizers, implemented with torch.linalg

#%%-----------------------------------------------------------------------------
#----------------------------------IMAGE MASK-----------------------------------
#%%-----------------------------------------------------------------------------
def getMask(C, threshold = 1e-5):
    '''For masking corrupted data to match masking of estimated coil profiles'''
    C_n = C[0,...] #extract a single coil profile
    mask = torch.zeros(C_n.shape, dtype = torch.float32)
    mask[torch.abs(C_n)>threshold] = 1
    return mask

#%%-----------------------------------------------------------------------------
#----------------------------------NAIVE RECON----------------------------------
#%%-----------------------------------------------------------------------------
def NaiveCoilCombo(s_data, U_temp, C):
   # IFFT and naive coil combination for a segment of k-space
   m_corrupted_temp = eop._ifft(s_data*U_temp, axes = (1,2,3))
   return torch.sum(torch.conj(C) * m_corrupted_temp, axis = 0)

def SubReconStore(s_data, U, C):
   m_store = []
   for i in range(U.shape[0]):
      print("TR {}".format(i+1), end='\r')
      m_temp = NaiveCoilCombo(s_data, U[i], C)
      m_store.append(m_temp)
   return m_store

#%%-----------------------------------------------------------------------------
#--------------------------------IMAGE ESTIMATION-------------------------------
#%%-----------------------------------------------------------------------------

def ImageEstimation(s_corrupted,C,U,T_CG,R_CG,res,CG_params):
    """
    CG-SENSE image reconstruction
    Based on scipy implementation
    """
    mask, maxiter, batch, CG_result = CG_params
    x0 = torch.zeros(s_corrupted.shape[1:], dtype=s_corrupted.dtype).to('cuda')
    if mask == None:
        mask = torch.ones(x0.shape, dtype=x0.dtype).to('cuda')
    #
    m_store = []
    r_store = []
    b = eop._EH(s_corrupted,C,U,T_CG,R_CG,res) #use default mode = "FFT"
    r0 = (b - eop._EH_E(x0,C,U,T_CG,R_CG,res))*mask #use default mode = "FFT"
    p0 = r0
    gamma0 = torch.vdot(r0.flatten(), r0.flatten())
    k_ = 0
    r_store.append(torch.norm(r0, p=2))
    m_store.append(b)
    #-------------------------------------
    def body_fun(vals):
        x, r, gamma, p, k, pAp, alpha, beta_ = vals
        Ap = eop._EH_E(p,C,U,T_CG,R_CG,res) #use default mode = "FFT"
        pAp = torch.vdot(p.flatten(), Ap.flatten())
        alpha = gamma / pAp
        x_ = x + alpha*p
        r_ = r - alpha*Ap
        gamma_ = torch.vdot(r_.flatten(), r_.flatten())
        beta_ = gamma_ / gamma
        p_ = (r_ + beta_ * p)*mask
        #
        m_store.append(x_)
        r_store.append(torch.norm(r_, p=2))
        return x_, r_, gamma_, p_, k + 1, pAp, alpha, beta_
    #-------------------------------------
    vals = (x0, r0, gamma0, p0, k_, 0, 0, 0)
    for steps in range(maxiter):
        print("CG iteration: {} out of {}".format(steps+1, maxiter))
        vals = body_fun(vals)
    #
    if CG_result == "optimal":
        min_ind = torch.argmin(torch.tensor(r_store))
        r_out = r_store[min_ind]
        m_out = m_store[min_ind]
    elif CG_result == "final":
        min_ind = -1
        r_out = r_store[min_ind]
        m_out = m_store[min_ind]
    return m_out, r_out, min_ind


#%%-----------------------------------------------------------------------------
#-------------------------------MOTION ESTIMATION-------------------------------
#%%-----------------------------------------------------------------------------

def _NNParam(x_init, grad = True):
    return torch.nn.Parameter(data=x_init, requires_grad=grad).to('cuda')

# def MotionEstimation_Adam(m,C,U,s_corrupted,T,R,res,ME_params):
#     class MotionModel(torch.nn.Module):
#         def __init__(self, T, R, res):
#             super(MotionModel, self).__init__()
#             self.Tx = _NNParam(T[0]); self.Ty = _NNParam(T[1]); self.Tz = _NNParam(T[2])
#             self.Rx = _NNParam(R[0]); self.Ry = _NNParam(R[1]); self.Rz = _NNParam(R[2])
#             self.res = res
#         #
#         def forward(self, m, C, U, inds): #shotwise evaluation
#             s_new = eop._E(m,C,U[inds,...],\
#                             [self.Tx[inds],self.Ty[inds],self.Tz[inds]],\
#                             [self.Rx[inds],self.Ry[inds],self.Rz[inds]],\
#                             self.res)
#             return s_new
#         #
#         def inverse(self, s, C, U, inds): #shotwise evaluation
#             m_new = eop._EH(s,C,U[inds,...],\
#                             [self.Tx[inds],self.Ty[inds],self.Tz[inds]],\
#                             [self.Rx[inds],self.Ry[inds],self.Rz[inds]],\
#                             self.res)
#             return m_new
#         #
#     #
#     optimizer, f_loss, maxiter, est_batch, stepsize = ME_params
#     model = MotionModel(T,R,res)
#     est_inds = torch.split(torch.arange(1,len(U)), est_batch, dim=0) #skip first shot
#     if optimizer == "Adam":
#         optim = torch.optim.Adam(model.parameters(), lr=stepsize)
#         for i in range(maxiter):
#             print("Adam iteration {} of {}".format(i+1, maxiter))
#             for inds in est_inds:
#                 print("Estimating shots {}".format(inds), end = '\r')
#                 s_est = model.forward(m, C, U, inds)
#                 U_temp = torch.sum(U[inds,...], axis = 0)
#                 if f_loss == "DC":
#                     DC_temp = s_corrupted*U_temp - s_est
#                     loss = torch.norm(DC_temp, p=2)**2
#                 elif f_loss == "DC_full": #ie. reintegrating target shots into rest of s_corrupted
#                     U_inv = abs(U_temp - 1)
#                     s_temp = s_est + s_corrupted*U_inv
#                     # DC_temp = s_corrupted - s_temp
#                     DC_temp = eop._ifft(s_corrupted, (1,2,3)) - eop._ifft(s_temp, (1,2,3))
#                     loss = torch.norm(DC_temp, p=2)**2
#                 elif f_loss == "GE":
#                     U_inv = abs(U_temp - 1)
#                     s_temp = s_est + s_corrupted*U_inv
#                     m_est = torch.sum(torch.conj(C)*eop._ifft(s_temp, (1,2,3)), axis = 0)
#                     loss = sum(mtc.GradientEntropy(m_est))
#                 elif f_loss == "DC+GE":
#                     DC_temp = s_corrupted*U_temp - s_est
#                     U_inv = abs(U_temp - 1)
#                     s_temp = s_est + s_corrupted*U_inv
#                     m_est = torch.sum(torch.conj(C)*eop._ifft(s_temp, (1,2,3)), axis = 0)
#                     loss = torch.norm(DC_temp, p=2)**2 + sum(mtc.GradientEntropy(m_est))
#                 loss.backward()
#                 optim.step()
#                 optim.zero_grad() #reset gradients
#             #
#     return model.Tx, model.Ty, model.Tz, model.Rx, model.Ry, model.Rz

def TR_2_Mtraj(T, R):
    #Convert between list of individual DOFs and combined array
    Tx, Ty, Tz = T; Rx, Ry, Rz = R
    Mtraj_out = torch.cat((Tx[:,None], Ty[:,None], Tz[:,None], \
                            Rx[:,None], Ry[:,None], Rz[:,None]), axis = 1)
    return Mtraj_out

def Mtraj_2_TR(Mtraj):
    #Convert between combined array and list of individual DOFs
    T = [Mtraj[:,0], Mtraj[:,1], Mtraj[:,2]]
    R = [Mtraj[:,3], Mtraj[:,4], Mtraj[:,5]]
    return T, R

def MotionEstimation(m,C,U,s_corrupted,T,R,res,ME_params):
    '''
    Estimating 3D motion parameters for Joint Estimation
    using BFGS (implemented from rfeinman/pytorch-minimize)
    '''
    def _DC_loss(Mtraj_est_inds, U_inds=None, m=None, C=None, res=None, s_corrupted=None):
        '''
        Data consistency loss for estimating Mtraj_est_inds (dims: [NSHOTS_SUBSET, 6]) 
        #
        Specifying kwargs to enable use with functools.partial
        '''
        T_est, R_est = Mtraj_2_TR(Mtraj_est_inds)
        s_est = eop._E(m,C,U_inds,T_est,R_est,res)
        U_temp = torch.sum(U_inds, axis = 0)
        DC_temp = s_corrupted*U_temp - s_est
        return torch.norm(DC_temp, p=2)**2
    #
    optimizer, f_loss, maxiter, est_batch, stepsize = ME_params
    options = {'lr':stepsize}
    Mtraj_est = TR_2_Mtraj(T, R) #initial motion estimates, convert to combined array
    #
    est_inds = torch.split(torch.arange(1,len(U)), est_batch, dim=0) #skip first shot
    if optimizer == "BFGS":
        for inds in est_inds:
            print("Estimating shots {}".format(inds), end = '\r')
            _f = partial(_DC_loss, U_inds=U[inds,...], m=m, C=C, res=res, s_corrupted=s_corrupted)
            result = torchmin.minimize(_f, Mtraj_est[inds,...], \
                                        method='bfgs', max_iter=maxiter, options=options)
            Mtraj_est[inds,...] = result['x'] #update motion parameter array
        #
    #
    T_est, R_est = Mtraj_2_TR(Mtraj_est)
    return T_est[0], T_est[1], T_est[2], R_est[0], R_est[1], R_est[2]


#%% ----------------------------------------------------------------------------
# -------------------------------JOINT ESTIMATION-------------------------------
# ---------------------------MULTI-LEVEL OPTIMIZATION---------------------------
# ------------------------------------------------------------------------------

def check_window(DC_grad, g_tol=1e-1, g_window=10):
    conv_flag = 0
    conv_iter = 0
    checks = DC_grad.shape[0] - g_window
    while not conv_flag and conv_iter < checks:
        temp_window = DC_grad[conv_iter:conv_iter+g_window]
        conv_flag = np.all(abs(temp_window.flatten()) < g_tol)
        conv_iter_out = conv_iter
        conv_iter += 1
    return conv_iter_out, conv_flag

def conv_condition(DC_store, g_window=10):
    DC_init = DC_store[0]
    DC_store_rescale = DC_store / DC_init
    DC_grad = DC_store_rescale[1:] - DC_store_rescale[:-1]
    #
    order = np.floor(np.log(1/DC_init) / np.log(10))
    g_tol = 10**(order)
    #
    conv_iter_out, conv_flag = check_window(DC_grad, g_tol=g_tol, g_window=g_window)
    return conv_flag, conv_iter_out

def UNet_magnitude(m_est, UNet_params):
    '''
    Running Stacked UNets with Self-Assisted Priors (Al-Masni et al 2022)
    Trained on magnitude data
    '''
    wpath, pads = UNet_params
    m_cnn_in = tf.convert_to_tensor(m_est.detach().cpu())
    m_cnn_out_mag = cnn.main(np.abs(m_cnn_in), pads, wpath + r'/magnitude')
    m_est = torch.from_numpy(m_cnn_out_mag).to('cuda')
    T = None; R = None; loss_store = None
    return m_est, T, R, loss_store


def UNet_complex(m_est, UNet_params):
    '''
    Running Stacked UNets with Self-Assisted Priors (Al-Masni et al 2022)
    Trained on real and imaginary components, separately
    '''
    wpath, pads = UNet_params
    m_cnn_in = tf.convert_to_tensor(m_est.detach().cpu())
    m_cnn_out_real = cnn.main(np.real(m_cnn_in), pads, wpath + r'/real')
    m_cnn_out_imag = cnn.main(np.imag(m_cnn_in), pads, wpath + r'/imag')
    m_est = torch.from_numpy(m_cnn_out_real + 1j*m_cnn_out_imag).to('cuda')
    T = None; R = None; loss_store = None
    return m_est, T, R, loss_store


def JE(m_est, s_corrupted, C, U, T, R, res, JE_params):
    '''
    Implementation of 3D Joint Estimation algorithm
    '''
    ME_params, CG_params, JE_maxiters, g_window = JE_params
    loss_store = []
    loss_store.append(torch.norm(s_corrupted-eop._E(m_est,C,U,T,R,res), p=2)**2)
    i = 0
    conv_flag = 0
    while i < JE_maxiters and not conv_flag:
        print("JE Iteration {} of {}".format(i+1, JE_maxiters))
        t1 = time()
        updates = MotionEstimation(m_est,C,U,s_corrupted,T,R,res,ME_params)
        #
        T_CG = [updates[0].detach(), updates[1].detach(), updates[2].detach()]
        R_CG = [updates[3].detach(), updates[4].detach(), updates[5].detach()]
        m_est, _, _ = ImageEstimation(s_corrupted,C,U,T_CG,R_CG,res,CG_params)
        #
        loss_store.append(torch.norm(s_corrupted-eop._E(m_est,C,U,T_CG,R_CG,res), p=2)**2)
        t2 = time()
        print("JE runtime: {} sec".format(t2 - t1))
        if i >= g_window:
            conv_flag, _ = conv_condition(loss_store, g_window)
        i+=1
    return m_est, T_CG, R_CG, loss_store


def UNetJE(m_est, s_corrupted, C, U, T, R, res, JE_params, UNet_params):
    '''
    Implementation of UNet-Assisted 3D Joint Estimation algorithm (Nghiem et al 2025) 
    '''
    ME_params, CG_params, JE_maxiters, g_window = JE_params
    loss_store = []
    loss_store.append(torch.norm(s_corrupted-eop._E(m_est,C,U,T,R,res), p=2)**2)
    i = 0
    conv_flag = 0
    while i < JE_maxiters and not conv_flag:
        print("JE Iteration {} of {}".format(i+1, JE_maxiters))
        t1 = time()
        m_cnn_out = UNet_complex(m_est, UNet_params)[0]
        #
        updates = MotionEstimation(m_cnn_out,C,U,s_corrupted,T,R,res,ME_params)
        #
        T_CG = [updates[0].detach(), updates[1].detach(), updates[2].detach()]
        R_CG = [updates[3].detach(), updates[4].detach(), updates[5].detach()]
        m_est, _, _ = ImageEstimation(s_corrupted,C,U,T_CG,R_CG,res,CG_params)
        #
        loss_store.append(torch.norm(s_corrupted-eop._E(m_est,C,U,T_CG,R_CG,res), p=2)**2)
        t2 = time()
        print("JE runtime: {} sec".format(t2 - t1))
        if i >= g_window:
            conv_flag, _ = conv_condition(loss_store, g_window)
        i+=1
    return m_est, T_CG, R_CG, loss_store



#%%-----------------------------------------------------------------------------
#-------------------------------MOTION ESTIMATION-------------------------------
#--------------------------------------UNET-------------------------------------
#%%-----------------------------------------------------------------------------

def ShotRegistration(m,C,U,s_corrupted,T,R,res,ME_params):
    '''
    Implementing algorithm to sequentially register segments of k-space
    in center-out fashion, such that overall image quality is improved.

    Conceptually, similar to autofocusing, albeit in image domain
    '''
    class MotionModel(torch.nn.Module):
        def __init__(self, T, R, res):
            super(MotionModel, self).__init__()
            self.Tx = T[0]; self.Ty = T[1]; self.Tz = T[2]
            self.Rx = R[0]; self.Ry = R[1]; self.Rz = R[2]
            self.res = res
        #
        def _STN(self, theta, m_MOVE_store):
            #Applying motion parameters to subdivisions of m_MOVE_store
            theta_new = theta.reshape((self.N_SHOTS, 6))
            m_out = torch.zeros(m_MOVE_store[0].shape, dtype = m_MOVE_store[0].dtype, device='cuda')
            for i in range(self.N_SHOTS):
                # print("Segment {}".format(i+1))
                m_temp = m_MOVE_store[i]
                T_temp = [theta_new[i,0],theta_new[i,1],theta_new[i,2]]
                R_temp = [theta_new[i,3],theta_new[i,4],theta_new[i,5]]
                TRm = eop.Translate_Regrid(m_temp, T_temp, self.res, mode='inv') #undo translation
                m_out += eop.Rotate_Regrid(TRm, R_temp, self.res, pad=(0,0,0), mode='inv') #undo rotation
            return m_out
        #
        def forward(self, m, C, U, inds): #shotwise evaluation
            s_new = eop._E(m,C,U[inds,...],\
                            [self.Tx[inds],self.Ty[inds],self.Tz[inds]],\
                            [self.Rx[inds],self.Ry[inds],self.Rz[inds]],\
                            self.res)
            return s_new
        #
        def inverse(self, s, C, U, inds): #shotwise evaluation
            m_new = eop._EH(s,C,U[inds,...],\
                            [self.Tx[inds],self.Ty[inds],self.Tz[inds]],\
                            [self.Rx[inds],self.Ry[inds],self.Rz[inds]],\
                            self.res)
            return m_new
        #
    #
    optimizer, f_loss, maxiter, est_batch, stepsize = ME_params
    model = MotionModel(T,R,res)
    est_inds = torch.split(torch.arange(1,len(U)), est_batch, dim=0) #skip first shot
    if optimizer == "Adam":
        optim = torch.optim.Adam(model.parameters(), lr=stepsize)
        for i in range(maxiter):
            print("Adam iteration {} of {}".format(i+1, maxiter))
            for inds in est_inds:
                print("Estimating shots {}".format(inds), end = '\r')
                s_est = model.forward(m, C, U, inds)
                U_temp = torch.sum(U[inds,...], axis = 0)
                if f_loss == "DC":
                    DC_temp = s_corrupted*U_temp - s_est
                    loss = torch.norm(DC_temp, p=2)**2
                elif f_loss == "DC_full":
                    U_inv = abs(U_temp - 1)
                    s_temp = s_est + s_corrupted*U_inv
                    # DC_temp = s_corrupted - s_temp
                    DC_temp = eop._ifft(s_corrupted, (1,2,3)) - eop._ifft(s_temp, (1,2,3))
                    loss = torch.norm(DC_temp, p=2)**2
                elif f_loss == "GE":
                    U_inv = abs(U_temp - 1)
                    s_temp = s_est + s_corrupted*U_inv
                    m_est = torch.sum(torch.conj(C)*eop._ifft(s_temp, (1,2,3)), axis = 0)
                    loss = sum(mtc.GradientEntropy(m_est))
                elif f_loss == "DC+GE":
                    DC_temp = s_corrupted*U_temp - s_est
                    U_inv = abs(U_temp - 1)
                    s_temp = s_est + s_corrupted*U_inv
                    m_est = torch.sum(torch.conj(C)*eop._ifft(s_temp, (1,2,3)), axis = 0)
                    loss = torch.norm(DC_temp, p=2)**2 + sum(mtc.GradientEntropy(m_est))
                loss.backward()
                optim.step()
                optim.zero_grad() #reset gradients
            #
    return model.Tx, model.Ty, model.Tz, model.Rx, model.Ry, model.Rz



#%%-----------------------------------------------------------------------------
#--------------------------------------UNET-------------------------------------
#%%-----------------------------------------------------------------------------

def rescale_sym(x, max):
	#For data ranging from [-max, max]
	#Output rescaled to [0,1]
	return (x + max) / (2*max) #RESCALE to [0,1]

def unscale_sym(x, max):
    return x*(2*max) - max

def UNet_Mag(m_est, trans_axes, pads, wpath, mask, cnn): #Magnitude Only
    m_cnn_in_init = np.transpose(m_est, axes=trans_axes[:3])
    m_cnn_in = rotate(m_cnn_in_init, angle=trans_axes[3], axes=(0,1))
    m_cnn_out_mag = cnn.main(np.abs(m_cnn_in), pads, wpath + r'/magnitude') #MAGNITUDE UNET
    m_cnn_out = m_cnn_out_mag
    #
    m_est_cnn_init = rotate(m_cnn_out, angle=-trans_axes[3], axes=(0,1))
    m_est_cnn = np.transpose(m_est_cnn_init, axes=trans_axes[:3])*mask
    m_est = m_est_cnn
    return m_est

def UNet_ReIm(m_est, trans_axes, pads, wpath, mask, cnn): #Real and Imaginary UNets, run sequentially
    m_cnn_in_init = np.transpose(m_est, axes=trans_axes[:3])
    m_cnn_in = rotate(m_cnn_in_init, angle=trans_axes[3], axes=(0,1))
    #
    m_cnn_out_real = cnn.main(np.real(m_cnn_in), pads, wpath + r'/real')
    m_cnn_out_imag = cnn.main(np.imag(m_cnn_in), pads, wpath + r'/imag')
    m_cnn_out = m_cnn_out_real + 1j*m_cnn_out_imag
    #
    m_est_cnn_init = rotate(m_cnn_out, angle=-trans_axes[3], axes=(0,1))
    m_est_cnn = np.transpose(m_est_cnn_init, axes=trans_axes[:3])*mask
    m_est = m_est_cnn
    return m_est

