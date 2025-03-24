"""
Defining optimizers used in image and motion estimation
"""

import numpy as np
import torch

from time import time
from functools import partial

import encode.encode_op as eop
import utils.metrics as mtc


#%%-----------------------------------------------------------------------------
#--------------------------------IMAGE ESTIMATION-------------------------------
#%%-----------------------------------------------------------------------------

def ConjugateGradient(state, params):
    """
    Based on scipy implementation

    For use in eop.ImageEstimation
    """
    x, r, gamma, p, k, pAp, alpha, beta_ = state
    C,U,T,R,res,mask = params #constants during image estimation
    #
    Ap = eop._EH_E(p,C,U,T,R,res) #use default mode = "FFT"
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

