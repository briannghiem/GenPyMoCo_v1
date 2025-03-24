"""
Defining encoding operators
For UNet-Assisted Joint Motion and Image Estimation
"""

# import jax.numpy as xp
import numpy as np
import torch

torch.pi = torch.tensor(np.pi).to('cuda') #for convenience

#%%-----------------------------------------------------------------------------
#--------------------------------HELPER FUNCTIONS-------------------------------
#-------------------------------FOR FFT OPERATORS-------------------------------
#%%-----------------------------------------------------------------------------

def _fft(input, axes):
    in_shift = torch.fft.ifftshift(input, dim = axes)
    in_fft = torch.fft.fftn(in_shift, dim = axes, norm = "ortho")
    return torch.fft.fftshift(in_fft, dim = axes).to('cuda')

def _ifft(input, axes):
    in_shift = torch.fft.ifftshift(input, dim = axes)
    in_ifft = torch.fft.ifftn(in_shift, dim = axes, norm = "ortho")
    return torch.fft.fftshift(in_ifft, dim = axes).to('cuda')

def _farray(ishape, res, axis):
    '''Compute kspace coordinates along axis'''
    return torch.fft.fftshift(torch.fft.fftfreq(ishape[axis], d = res[axis]))

def _fgrid(ishape, res):
    '''Compute grid of kspace coordinates'''
    fx_array = _farray(ishape, res, 0)
    fy_array = _farray(ishape, res, 1)
    fz_array = _farray(ishape, res, 2)
    #NB. Current torch version doesn't have indexing
    #However, default index is 'ij', so no need to swap axes
    fx_grid, fy_grid, fz_grid = torch.meshgrid(fx_array, fy_array, fz_array)
    return fx_grid.to('cuda'), fy_grid.to('cuda'), fz_grid.to('cuda')

def _iarray(ishape, res, axis):
    '''Compute image-space coordinates along axis'''
    return torch.arange(-ishape[axis]//2, ishape[axis]//2)*res[axis]

def _igrid(ishape, res):
    '''Compute image-space coordinates'''
    x_array = _iarray(ishape, res, 0)
    y_array = _iarray(ishape, res, 1)
    z_array = _iarray(ishape, res, 2)
    #NB. Current torch version doesn't have indexing
    #However, default index is 'ij', so no need to swap axes
    x_grid, y_grid, z_grid = torch.meshgrid(x_array, y_array, z_array)
    return x_grid.to('cuda'), y_grid.to('cuda'), z_grid.to('cuda')


#%%-----------------------------------------------------------------------------
#--------------------------------MOTION OPERATORS-------------------------------
#-----------------------------------USING FFT-----------------------------------
#%%-----------------------------------------------------------------------------

#Helper functions for translation
def _phaseRamp(D, fgrid, axis):
    '''Compute phase ramp for given axis'''
    phase = fgrid[axis] * D[axis]
    pi = torch.tensor(torch.pi)
    return torch.exp(-2j*pi*phase).to('cuda')

def _trans1D(D, fgrid, axis, input):
    '''Apply translation (phase ramp) for given axis'''
    ramp = _phaseRamp(D, fgrid, axis)
    ft = _fft(input, (axis,))
    pm = ft * ramp
    out = _ifft(pm, (axis,))
    return out

#-------------------------------------------------------------------------------
def Translate(input, D, res, mode='fwd'):
    '''Implementing 3D Translation via k-space Linear Phase Ramps (Cordero-Grande et al, 2016)'''
    fgrid = _fgrid(input.shape, res)
    if mode == 'inv':
        D = [-d for d in D]
    Tx = _trans1D(D, fgrid, 0, input)
    TyTx = _trans1D(D, fgrid, 1, Tx)
    TzTyTx = _trans1D(D, fgrid, 2, TyTx)
    return TzTyTx

#Helper functions for rotation
def _pad(input, pad):
    '''Add symmetric padding to input'''
    output = torch.nn.functional.pad(input, \
                                     (pad[0],pad[0],\
                                      pad[1],pad[1],\
                                        pad[2],pad[2])) #dif syntax compared to numpy.pad
    return output

def _unpad_inds(pad):
    '''Output start and stop slice indices for unpad'''
    start = pad
    if pad == 0:
        stop = None
    else:
        stop = -pad
    return start, stop

def _unpad(input, pad):
    '''Remove input's symmetric padding'''
    indx_start, indx_stop = _unpad_inds(pad[0])
    indy_start, indy_stop = _unpad_inds(pad[1])
    indz_start, indz_stop = _unpad_inds(pad[2])
    output = input[indx_start:indx_stop, indy_start:indy_stop, indz_start:indz_stop]
    return output

def _deg2rad(val): #convert to rad
    pi = torch.tensor(torch.pi)
    return val * (pi / 180)

def _phase_tan(R_i, fgrid_i, igrid_i):
    phase = -torch.tan(_deg2rad(R_i/2)) * torch.multiply(fgrid_i, igrid_i)
    pi = torch.tensor(torch.pi)
    return torch.exp(-2j*pi*phase).to('cuda')

def _phase_sin(R_i, fgrid_i, igrid_i):
    phase = torch.sin(_deg2rad(R_i)) * torch.multiply(fgrid_i, igrid_i)
    pi = torch.tensor(torch.pi)
    return torch.exp(-2j*pi*phase).to('cuda')

def _shear_tan(R_i, fgrid_i, igrid_i, tan_axis, input):
    #Compute nonlinear phase ramp for shearing along given axis
    phase = _phase_tan(R_i, fgrid_i, igrid_i)
    ft = _fft(input, (tan_axis,))
    pm = ft * phase
    out = _ifft(pm, (tan_axis,))
    return out

def _shear_sin(R_i, fgrid_i, igrid_i, sin_axis, input):
    #Compute nonlinear phase ramp for shearing along given axis
    phase = _phase_sin(R_i, fgrid_i, igrid_i)
    ft = _fft(input, (sin_axis,))
    pm = ft * phase
    out = _ifft(pm, (sin_axis,))
    return out

def _rot1D(R, axis, fgrids, igrids, axes, input): #3-pass shear decomposition
    R_i = R[axis]
    S_tan1 = _shear_tan(R_i, fgrids[axes[0]], igrids[axes[1]], axes[0], input)
    S_sin = _shear_sin(R_i, fgrids[axes[1]], igrids[axes[0]], axes[1], S_tan1)
    S_tan2 = _shear_tan(R_i, fgrids[axes[0]], igrids[axes[1]], axes[0], S_sin)
    return S_tan2

def Rotate(input, R, res, pad=(0,0,0), mode='fwd'):
    '''Implementing 9-Pass Shear Decomposition of 3D Rotation (Unser et al, 1995)'''
    m_pad = _pad(input, pad)
    fgrids = _fgrid(m_pad.shape, res)
    igrids = _igrid(m_pad.shape, res)
    if mode=='fwd':
        Rx = _rot1D(R, 0, fgrids, igrids, [1,2], m_pad)
        RyRx = _rot1D(R, 1, fgrids, igrids, [2,0], Rx)
        RzRyRx = _rot1D(R, 2, fgrids, igrids, [0,1], RyRx)
        out = _unpad(RzRyRx, pad)
    elif mode=='inv': #reverse order of rotation application (rotations aren't commutative)
        R = [-r for r in R]
        Rz = _rot1D(R, 2, fgrids, igrids, [0,1], m_pad)
        RyRz = _rot1D(R, 1, fgrids, igrids, [2,0], Rz)
        RxRyRz = _rot1D(R, 0, fgrids, igrids, [1,2], RyRz)
        out = _unpad(RxRyRz, pad)
    #
    return out


#%%-----------------------------------------------------------------------------
#-------------------------------ENCODING OPERATORS------------------------------
#-----------------------------------USING FFT-----------------------------------
#%%-----------------------------------------------------------------------------
def _E_n(m_in,C,U_n,T_n,R_n,res,method='FFT'): #Apply forward encoding operator for single shot
    #Apply FWD encoding
    if method=='FFT':
        Rm = Rotate(m_in, R_n, res, pad=(0,0,0), mode='fwd') #pad only for larger rotations
        TRm = Translate(Rm, T_n, res, mode='fwd')
    elif method=='grid':
        Rm = Rotate_Regrid(m_in, R_n, res, pad=(0,0,0), mode='fwd') #pad only for larger rotations
        TRm = Translate_Regrid(Rm, T_n, res, mode='fwd')
    CTRm = C * TRm
    FCTRm = _fft(CTRm, (1,2,3))
    s_n = U_n * FCTRm
    return s_n

def _EH_n(s_in,C,U_n,T_n,R_n,res,method='FFT'): #Apply inverse encoding operator for single shot
    #Apply INV encoding
    Us = U_n*s_in
    FUs = _ifft(Us, (1,2,3))
    CFUs = torch.sum(torch.conj(C) * FUs, axis = 0)
    if method=='FFT':
        TCFUs = Translate(CFUs, T_n, res, mode='inv')
        m_n = Rotate(TCFUs, R_n, res, pad=(0,0,0), mode='inv') #pad only for larger rotations
    elif method=='grid':
        TCFUs = Translate_Regrid(CFUs, T_n, res, mode='inv')
        m_n = Rotate_Regrid(TCFUs, R_n, res, pad=(0,0,0), mode='inv') #pad only for larger rotations
    return m_n

def _E(m_in,C,U,T,R,res,method='FFT'):
    #Create stacked affine transforms
    s_out = torch.zeros(C.shape, dtype = C.dtype).to('cuda')
    Tx,Ty,Tz = T; Rx,Ry,Rz = R
    #Create chunks based on batch size
    batch = 1 #force sequential evaluation
    Tx_batch = torch.split(Tx, batch)
    Ty_batch = torch.split(Ty, batch)
    Tz_batch = torch.split(Tz, batch)
    Rx_batch = torch.split(Rx, batch)
    Ry_batch = torch.split(Ry, batch)
    Rz_batch = torch.split(Rz, batch)
    U_batch = torch.split(U, batch, dim=0)
    #
    for n in range(len(U_batch)): #batch of motion states
        print("Batch {} of {}".format(n+1, len(U_batch)), end='\r')
        R_n = [Rx_batch[n], Ry_batch[n], Rz_batch[n]]
        T_n = [Tx_batch[n], Ty_batch[n], Tz_batch[n]]
        s_out += _E_n(m_in,C,U_batch[n],T_n,R_n,res,method)
    return s_out

def _EH(s_in,C,U,T,R,res,method='FFT'):
    m_out = torch.zeros(s_in.shape[1:], dtype=C.dtype).to('cuda')
    #Create stacked affine transforms
    Tx,Ty,Tz = T; Rx,Ry,Rz = R
    #Create chunks based on batch size
    batch = 1 #force sequential evaluation
    Tx_batch = torch.split(Tx, batch)
    Ty_batch = torch.split(Ty, batch)
    Tz_batch = torch.split(Tz, batch)
    Rx_batch = torch.split(Rx, batch)
    Ry_batch = torch.split(Ry, batch)
    Rz_batch = torch.split(Rz, batch)
    U_batch = torch.split(U, batch, dim=0)
    #
    for n in range(len(U_batch)): #batch of motion states
        print("Batch {} of {}".format(n+1, len(U_batch)), end='\r')
        R_n = [Rx_batch[n], Ry_batch[n], Rz_batch[n]]
        T_n = [Tx_batch[n], Ty_batch[n], Tz_batch[n]]
        m_out += _EH_n(s_in,C,U_batch[n],T_n,R_n,res,method)
    return m_out

def _EH_E(m_in,C,U,T,R,res,method='FFT'):
    s_temp = _E(m_in,C,U,T,R,res,method)
    m_out = _EH(s_temp,C,U,T,R,res,method)
    return m_out



#%%-----------------------------------------------------------------------------
#-------------------------------ENCODING OPERATORS------------------------------
#--------------------------------USING REGRIDING--------------------------------
#%%-----------------------------------------------------------------------------

import torch
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode

#Translations
def Tx_Regrid(m, D):
    m = m.permute((2, 1, 0))
    Tx = affine(m, interpolation=InterpolationMode.BILINEAR, angle=0, \
        translate=[float(-D[0]), 0], scale = 1.0, shear = 0, fill=0.0)
    Tx = Tx.permute((2, 1, 0))
    return Tx

def Ty_Regrid(m, D):
    m = m.permute((2, 1, 0))
    Ty = affine(m, interpolation=InterpolationMode.BILINEAR, angle=0, \
        translate=[0, float(-D[1])], scale = 1.0, shear = 0, fill=0.0)
    Ty = Ty.permute((2, 1, 0))
    return Ty

def Tz_Regrid(m, D):
    Tz = affine(m, interpolation=InterpolationMode.BILINEAR, angle=0, \
        translate=[float(-D[2]), 0], scale = 1.0, shear = 0, fill=0.0)
    return Tz

def Translate_Regrid(input, D, res, mode='fwd'):
    D = [-d for d in D]#need to reverse orientation to match to FFT implementation
    if mode == 'inv':
        D = [-d for d in D]
    #
    Tx = Tx_Regrid(input, D)
    TyTx = Ty_Regrid(Tx, D)
    TzTyTx = Tz_Regrid(TyTx, D)
    #
    return TzTyTx.to('cuda')


#Rotations
def Rx_Regrid(m, R):
    Rx = affine(m, interpolation=InterpolationMode.BILINEAR, angle=float(R[0]), \
        translate=[0, 0], scale = 1.0, shear = 0, fill=0.0)
    return Rx

def Ry_Regrid(m, R):
    m = m.permute((1, 0, 2))
    Ry = affine(m, interpolation=InterpolationMode.BILINEAR, angle=float(-R[1]), \
        translate=[0, 0], scale = 1.0, shear = 0, fill=0.0)
    Ry = Ry.permute((1, 0, 2))
    return Ry

def Rz_Regrid(m, R):
    m = m.permute((2, 1, 0))
    Rz = affine(m, interpolation=InterpolationMode.BILINEAR, angle=float(-R[2]), \
        translate=[0, 0], scale = 1.0, shear = 0, fill=0.0)
    Rz = Rz.permute((2, 1, 0))
    return Rz

def Rotate_Regrid(input, R, res, pad=(0,0,0), mode='fwd'):
    '''Implementing 9-Pass Shear Decomposition of 3D Rotation (Unser et al, 1995)'''
    R = [-r for r in R] #need to reverse orientation to match to FFT implementation
    m_pad = _pad(input, pad)
    if mode=='fwd':
        Rx = Rx_Regrid(m_pad, R)
        RyRx = Ry_Regrid(Rx, R)
        RzRyRx = Rz_Regrid(RyRx, R)
        out = _unpad(RzRyRx, pad)
    elif mode=='inv': #reverse order of rotation application (rotations aren't commutative)
        R = [-r for r in R]
        Rz = Rz_Regrid(m_pad, R)
        RyRz = Ry_Regrid(Rz, R)
        RxRyRz = Rx_Regrid(RyRz, R)
        out = _unpad(RxRyRz, pad)
    #
    return out.to('cuda')
