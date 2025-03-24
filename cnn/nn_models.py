"""
Defining neural network models for motion estimation
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import encode.encode_op as eop
import utils.metrics as mtc

#%%-----------------------------------------------------------------------------
#---------------------------------Neural Network--------------------------------
#---------------------------Estimating Motion Parameters -----------------------
#%%-----------------------------------------------------------------------------

class Model_A(torch.nn.Module):
    '''
    Neural network for motion estimation
    NN architecture was inspired by Gu et al, IEEE 2019, Two-stage Unsupervised Learning [...]
    Code reworked from Strittmatter et al Med Phys 2024 DL-based affine medical image registration [...]
    '''
    def __init__(self, res = (1,1,1), kernel_size = 3, \
                 channels = [4,8,16,32,64], LeakyReLU_slope = 0.2):
        super(Model_A, self).__init__()
        self.res = res
        self.N_SHOTS = 128
        self.kernel_size = kernel_size
        self.channels = channels
        #
        self.MaxPool = nn.MaxPool3d(kernel_size=2)
        self.ReLU = nn.LeakyReLU(negative_slope=LeakyReLU_slope)
        #
        self.Conv1 = nn.Conv3d(in_channels=self.channels[0]//2, out_channels=self.channels[0], \
                                kernel_size=self.kernel_size, padding="same", device='cuda')
        #
        self.Conv2 = nn.Conv3d(in_channels=self.channels[0], out_channels=self.channels[1], \
                                kernel_size=self.kernel_size, padding="same", device='cuda')
        #
        self.Conv3 = nn.Conv3d(in_channels=self.channels[1], out_channels=self.channels[2], \
                                kernel_size=self.kernel_size, padding="same", device='cuda')
        #
        self.Conv4 = nn.Conv3d(in_channels=self.channels[2], out_channels=self.channels[3], \
                                kernel_size=self.kernel_size, padding="same", device='cuda')
        #
        self.Conv5 = nn.Conv3d(in_channels=self.channels[3], out_channels=self.channels[4], \
                                kernel_size=self.kernel_size, padding="same", device='cuda')
        #
        self.Flatten = nn.Flatten()
        self.FC = nn.Linear(in_features=24576, out_features=self.N_SHOTS*6, device='cuda')
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
    def _NN(self, m_in):
        '''
        Adapted from Gu et al, IEEE 2019
        5-layer regression CNN for estimating motion parameters between k-space segments
        '''
        m_temp = m_in
        m_temp = self.ReLU(self.MaxPool(self.Conv1(m_temp)))
        m_temp = self.ReLU(self.MaxPool(self.Conv2(m_temp)))
        m_temp = self.ReLU(self.MaxPool(self.Conv3(m_temp)))
        m_temp = self.ReLU(self.MaxPool(self.Conv4(m_temp)))
        m_temp = self.ReLU(self.MaxPool(self.Conv5(m_temp)))
        #
        theta = self.FC(self.Flatten(m_temp))
        return theta
    #
    def loss(self, theta, m_MOVE_store):
        '''
        Unsupervised loss; evaluating image gradient entropy of resulting image
        '''
        m_out = self._STN(theta, m_MOVE_store)
        return sum(mtc.GradientEntropy(m_out))
    #
    def forward(self, m_MOVE_store):
        '''
        Estimating motion parameters for each k-space segment directly
        In this implementation, the initial two channels separate the real and imaginary parts of the image data
        '''
        m_in = sum(m_MOVE_store)
        self.N_SHOTS = len(m_MOVE_store) #number of k-space segments
        m_in = torch.cat((torch.real(m_in)[None,...], \
                            torch.imag(m_in)[None,...]), axis = 0)[None,...] #1st index = #samples
        theta = self._NN(m_in)
        return theta
    #


'''
#---------------------------------------
#---------------------------------------
#Figuring out how to subdivide the 256 TRs along PE1 

def NParams_CNN(kw,kh,kl,Cin,Cout):
   return (kw*kh*kl*Cin + 1) * Cout

def NParams_FC(Cin, Cout):
   return (Cin + 1) * Cout

def NParams_GuNN(kw,kh,kl,N_TR):
    # dims0 = [2,256,256,384]
    dims0 = [2,256,256,192]
    #
    L1 = NParams_CNN(kw,kh,kl,dims0[0],dims0[0]*2)
    dims1 = [dims0[0]*2,dims0[1]//2,dims0[2]//2,dims0[3]//2]
    #
    L2 = NParams_CNN(kw,kh,kl,dims1[0],dims1[0]*2)
    dims2 = [dims1[0]*2,dims1[1]//2,dims1[2]//2,dims1[3]//2]
    #
    L3 = NParams_CNN(kw,kh,kl,dims2[0],dims2[0]*2)
    dims3 = [dims2[0]*2,dims2[1]//2,dims2[2]//2,dims2[3]//2]
    #
    L4 = NParams_CNN(kw,kh,kl,dims3[0],dims3[0]*2)
    dims4 = [dims3[0]*2,dims3[1]//2,dims3[2]//2,dims3[3]//2]
    #
    L5 = NParams_CNN(kw,kh,kl,dims4[0],dims4[0]*2)
    dims5 = [dims4[0]*2,dims4[1]//2,dims4[2]//2,dims4[3]//2]
    #
    L6 = NParams_FC(torch.prod(torch.tensor(dims5)), 6*N_TR)
    #
    sum = L1 + L2 + L3 + L4 + L5 + L6
    print("Total # trainable parameters for {} TRs: {}".format(N_TR, sum))
    return sum

kw, kh, kl = (3,3,3)
subdiv = [64,32,16,8,4,2,1]
for val in subdiv:
   temp = NParams_GuNN(kw,kh,kl,val)

'''
