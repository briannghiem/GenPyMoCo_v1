"""
MAIN SCRIPT
Running UNet-Assisted Joint Motion and Image Estimation

Reproducing results for Figure 2 ("Large Motion Parameter")
"""

import os
import pathlib as plib
import jax.numpy as xp
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc
import motion.motion_sim as msi


def main(root, dpath, test_flag, case, motion_lv):
    #---------------------------------------------------------------------------
    #-----------------------Image Acquisition Simulation------------------------
    #---------------------------------------------------------------------------
    #Loading GT data
    m_GT_init = xp.load(dpath + r'/img_GT.npy') #initial orientation SI, LR, AP
    m_GT = xp.pad(m_GT_init[:,:,:,0,0] + 1j*m_GT_init[:,:,:,1,0], ((1,1), (0,0), (0,0)))
    del m_GT_init
    C = xp.load(dpath + r'/sens.npy')
    res = xp.array([1,1,1]) #image resolution: 1 mm iso
    #Transpose to reorient as LR, AP, SI
    m_GT = xp.transpose(m_GT, (1,2,0))
    m_GT = xp.abs(m_GT[6:-6, 3:-3, :])
    xp.save(dpath + r'/img_CG_mag.npy', m_GT)
    #
    mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask) #mask of coverage of coil profiles from BART estimation
    cerebrum_mask = xp.ones(m_GT.shape) #Set mask to identity for simulations; otherwise identify base of cerebellum
    #---------------------------------------------------------------------------
    #Generate sampling pattern with interleaved PE1
    Rs = 1
    TR_shot = 16
    U_array = xp.transpose(msi.make_samp(xp.transpose(m_GT, (1,0,2)), Rs, TR_shot, order='interleaved'), (0,2,1,3))
    U = eop._U_Array2List(U_array, m_GT.shape)    
    #---------------------------------------------------------------------------
    #Motion trajectory
    R_pad = (10, 10, 10)
    batch = 1
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
    #
    j = 1; k = 1 #legacy parameters, from training dataset script
    rand_keys = msi._gen_key(60+case, j, k)
    Mtraj_GT = msi._gen_traj(rand_keys, motion_lv, len(U), motion_specs)
    xp.save(dpath + r'/Mtraj.npy', Mtraj_GT)
    #
    s_corrupted = eop.Encode(m_GT, C, U, Mtraj_GT, res, batch=batch)
    xp.save(dpath + r'/s_corrupted_mag.npy', s_corrupted)
    #---------------------------------------------------------------------------
    #------------------JOINT IMAGE RECON AND MOTION ESTIMATION------------------
    #---------------------------------------------------------------------------
    #Initializing update vars
    Mtraj_init = xp.zeros((len(U), 6))
    Mtraj_est = Mtraj_init
    CG_maxiter = 3 #limit CG_iter to 3 iters for fully-sampled data to prevent artifacts
    ME_maxiter = 1 #motion estimation maxiter
    LS_maxiter = 20 #line search maxiter for BFGS algorithm
    CG_tol = 1e-7 #relative tolerance
    CG_atol = 1e-4 #absolute tolerance
    CG_mask = 0 #turn on for in-vivo dataset, turn off for simulated dataset
    grad_window = 10 #JE checks if loss gradient is below threshold within a window
    #Initialize stores
    m_loss_store = []
    m_cnn_store = []
    Mtraj_store = []
    DC_store = []
    #---------------------------------------------------------------------------
    #Reconstruct image via EH --> don't need CG algorithm since data is fully-sampled
    m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
    m_corrupted = m_init
    #----------------------------------------
    m_est_rmse = mtc.evalPE(m_corrupted, m_GT, mask)
    m_est_ssim = mtc.evalSSIM(m_corrupted, m_GT, mask=mask)
    m_loss_store.append([m_est_rmse, m_est_ssim])
    print("RMSE of Corrupted Image: {:.2f} %".format(m_est_rmse))
    print("SSIM of Corrupted Image: {}".format(m_est_ssim))
    m_est = m_corrupted
    #---------------------------------------------------------------------------
    #Loading trained UNet model
    #UNet takes in data as [LR, AP, SI]
    wpath = root + r'/cnn/weights'
    pads = [11,3]
    #---------------------------------------------------------------------------
    #Set up saving directory and JE parameters
    trans_axes = (0,1,2,0) 
    cnn_flag = test_flag[0]
    JE_flag = test_flag[1]
    if JE_flag and cnn_flag: #UNet + JE
        spath = dpath + r'/Method3'
        max_loops = 11
        if max_loops < grad_window:
            raise ValueError('Need to run for at least {} iterations'.format(grad_window))
    elif JE_flag and not cnn_flag: #JE
        max_loops = 11
        spath = dpath + r'/Method2'
        if max_loops < grad_window:
            raise ValueError('Need to run for at least {} iterations'.format(grad_window))
    elif not JE_flag and cnn_flag: #UNet
        max_loops = 1
        spath = dpath + r'/Method1'    
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #---------------------------------------------------------------------------
    #Package all algorithm parameters into lists
    dscale = 1
    continuity = 0
    JE_params = [max_loops, ME_maxiter, LS_maxiter, CG_maxiter, CG_tol, \
                CG_atol, CG_mask, batch, mask, continuity, grad_window]
    CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath]
    init_est = [m_est, Mtraj_est]
    fixed_vars = [m_init, s_corrupted, C, U, dscale, res, \
                    spath, m_GT, R_pad, cerebrum_mask]
    #
    #Compute initial total DC loss
    DC_update = rec._f(Mtraj_init, m_est=m_corrupted, C=C, res=res, \
                        U=U, R_pad=R_pad, s_corrupted=s_corrupted)
    DC_store.append(DC_update)
    xp.save(spath + r"/DC_store.npy", DC_store)
    #
    #Run motion correction
    stores = [m_cnn_store, Mtraj_store, m_loss_store, DC_store]
    m_est, m_loss_store, Mtraj_store, m_cnn_store, conv_result = rec.JointEst(init_est, fixed_vars, \
                                                                                stores, cnn, \
                                                                                CNN_params, JE_params)
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store, conv_result

#%% Run main()
if __name__ == "__main__":
    root = r'/home/nghiemb/CodeRepo'
    case = 1
    motion_lv = 'severe'
    dpath = root + r'/data/synthetic/{}/Sub{}'.format(motion_lv, case)
    test_flags = [[1,1], [0,1], [1,0]] #[CNN, JE]
    for test_flag in test_flags: #Iterating through the 3 methods 
        main_output = main(root, dpath, test_flag, case, motion_lv)
        spath, m_corrupted, m_final, m_loss_store, Mtraj_store, conv_result = main_output
        xp.save(spath + r"/m_corrupted.npy", m_corrupted)
        xp.save(spath + r"/m_final.npy", m_final)
        xp.save(spath + r"/m_loss_store.npy", m_loss_store)
        xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)
        xp.save(spath + r"/conv_result.npy", conv_result)
