import numpy as np
from scipy.interpolate import interp1d
from ..utils.constants import *

"""Cosmo quantities"""
def Ez_inv(z,H0,Om0):
    return 1/np.sqrt((1.-Om0) + Om0*np.power((1+z),3))

"""Approximate luminosity distance"""
#https://arxiv.org/pdf/1111.6396.pdf
def Phi(x):
    num = 1 + 1.320*x + 0.4415* np.power(x,2) + 0.02656*np.power(x,3)
    den = 1 + 1.392*x + 0.5121* np.power(x,2) + 0.03944*np.power(x,3)
    return num/den

def xx(z,Om0):
    return (1.0-Om0)/Om0/np.power(1.0+z,3)

def dL_approx(z,H0,Om0):
    D_H = (Clight/1.0e3)  / H0 #Mpc
    
    return 2.*D_H * (1.+z) * (Phi(xx(0.,Om0)) - Phi(xx(z,Om0))/np.sqrt(1.+z))/np.sqrt(Om0)

def z_at_dl_approx(dl,H0,Om0,zmin=1e-3,zmax=50):
    #dl in Mpc
    zs = np.logspace(np.log10(zmin),np.log10(zmax),1000)
    z_dl = interp1d(dL_approx(zs,H0,Om0),zs,bounds_error=False,fill_value='extrapolate')
    return z_dl(dl)

def diff_comoving_volume_approx(z,H0,Om0):
    dL = dL_approx(z,H0,Om0)#gwcosmo.d_L(z,H0,Om0) #Mpc
    Ez_i = Ez_inv(z,H0,Om0)
    D_H = (Clight/1e3)  / H0 #Mpc
    
    return 1.0e-9 * (4.*np.pi) * np.power(dL,2) * D_H * Ez_i / np.power(1.+z,2.)

def detector_to_source_frame(m1z,m2z,dL,H0,Om0,zmin=1e-3,zmax=100):
    z = z_at_dl_approx(dL,H0,Om0,zmin,zmax)
    m1 = m1z / (1. + z)
    m2 = m2z / (1. + z)
    return m1, m2, z