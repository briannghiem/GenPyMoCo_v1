# -*- coding: utf-8 -*-
"""
Created on Sun Mar 7 9:15:00 2021

@author: brian
"""
"""Simulating Motion"""

import numpy as np
import warnings


#%%-----------------------------------------------------------------------------
#--------------------------------SAMPLING PATTERN-------------------------------
#%%-----------------------------------------------------------------------------

def seq_order(U_sum,m,Rs,TR_shot,nshots):
    '''Sequential k-space sampling order'''
    U_seq = np.zeros((nshots, m.shape[0], m.shape[1], m.shape[2]))
    for i in range(nshots):
        ind_start = i*(Rs*TR_shot)
        if i == (nshots-1):
            ind_end = -1
        else:
            ind_end = (i+1)*(Rs*TR_shot)
        val = U_sum[ind_start:ind_end,...]
        U_seq[i,ind_start:ind_end,...] = val
        #
    #
    return U_seq

def int_order(U_sum,m,Rs,TR_shot,nshots):
    '''Interleaved k-space sampling order'''
    U_int = np.zeros((nshots, m.shape[0], m.shape[1], m.shape[2]))
    for i in range(nshots):
        interval = Rs*nshots
        ind_start = i*Rs
        ind_end = ind_start + TR_shot*interval
        for j in range(ind_start,ind_end,interval):
            try:
                U_int[i,j,:,:] = 1
            except:
                exit
        #
    #
    return U_int

def make_samp(m, Rs, TR_shot, order='interleaved', tile_dims = None):
    #Base sampling pattern
    try:
        R_PE1, R_PE2 = Rs
    except:
        R_PE1 = Rs
        R_PE2 = 1
    U_sum = np.zeros(m.shape)
    U_sum[::R_PE1,::R_PE2,:] = 1 #cumulative sampling, with R = 2
    nshots = int(np.round(m.shape[0]/(R_PE1*TR_shot)))
    #---------------------------------------------------------------------------
    #Generating different sampling orderings
    if order == "sequential":
        U = seq_order(U_sum, m, R_PE1, TR_shot, nshots)
    elif order == "interleaved":
        U = int_order(U_sum, m, R_PE1, TR_shot, nshots)
    else:
        warnings.warn("Error: sampling order not yet implemented; defaulting to sequential order")
        U = seq_order(U_sum, m, R_PE1, TR_shot, nshots)
    #
    return U

def _U_Array2List(U, m_shape):
    U_list = []
    for i in range(U.shape[0]):
        RO_temp = np.arange(0, m_shape[0])
        PE1_temp = np.where(U[i,0,:,0] == 1)[0]
        PE2_temp = np.arange(0, m_shape[2])
        U_list.append([RO_temp, PE1_temp, PE2_temp])
    return U_list

def _gen_U_n(U_vals, m_shape):
    #Lazy evaluation of sampling pattern
    U_RO = np.zeros(m_shape[0]); U_RO = U_RO.at[U_vals[0]].set(1) 
    U_PE1 = np.zeros(m_shape[1]); U_PE1 = U_PE1.at[U_vals[1]].set(1)
    U_PE2 = np.zeros(m_shape[2]); U_PE2 = U_PE2.at[U_vals[2]].set(1)    
    return np.multiply.outer(U_RO, np.outer(U_PE1, U_PE2))


#%%-----------------------------------------------------------------------------
#-------------------------------MOTION SIMULATION-------------------------------
#%%-----------------------------------------------------------------------------

def _gen_traj_dof(rand_key, motion_lv, dof, nshots, motion_specs):
    '''
    Input:
        rand_key=jax.random.PRNGKey object,
        motion_lv={'mild','moderate','severe','extreme'},
        dof={'Tx','Ty','Tz','Rx','Ry','Rz'}
        nshots=int # of motion states
    Output:
        np.array of motion trajectory for a given DOF
    '''
    p_val = motion_specs[motion_lv][dof][1]
    p_array = np.array([p_val/2, 1-p_val, p_val/2])
    opts = np.array([-1,0,1]) #move back, stay, move fwd
    maxval = motion_specs[motion_lv][dof][0]
    minval = maxval / 2
    rng = np.random.default_rng(seed=rand_key)
    array = rng.choice(a = opts, size=(nshots-1,), p = p_array) #binary array
    array = np.concatenate((np.array([0]), array)) #ensure first motion state is origin
    vals = rng.uniform(size=(nshots,),low=minval, high=maxval) #displacements
    return np.cumsum(array * vals) #absolute value of motion trajectory

def _gen_traj(rand_keys, motion_lv, nshots, motion_specs):
    '''
    Input:
        rand_key=jax.random.PRNGKey object,
        motion_lv={'mild','moderate','severe','extreme'},
        nshots=int # of motion states
    Output:
        np.array of motion trajectory across all 6 DOFs
    '''
    out_array = np.zeros((nshots, 6))
    out_array[:,0] = _gen_traj_dof(rand_keys[0], motion_lv, 'Tx', nshots, motion_specs)
    out_array[:,1] = _gen_traj_dof(rand_keys[1], motion_lv, 'Ty', nshots, motion_specs)
    out_array[:,2] = _gen_traj_dof(rand_keys[2], motion_lv, 'Tz', nshots, motion_specs)
    out_array[:,3] = _gen_traj_dof(rand_keys[3], motion_lv, 'Rx', nshots, motion_specs)
    out_array[:,4] = _gen_traj_dof(rand_keys[4], motion_lv, 'Ry', nshots, motion_specs)
    out_array[:,5] = _gen_traj_dof(rand_keys[5], motion_lv, 'Rz', nshots, motion_specs)
    return out_array

def _gen_seq(i,j,k,dof):
    a1 = (i+j+dof+1)**2 + (5*j)**2 + (17*i)**2 + (k*1206)**2 #including the exponent to guarantee different random value than training dataset
    return a1

def _gen_key(i, j, k):
    return [_gen_seq(i,j,k,dof) for dof in range(6)]
