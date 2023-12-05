#import numpyro
import jax.numpy as jnp
from ..utils.constants import *

def Ez_inv(z,Om0):
    return 1./jnp.sqrt((1.-Om0) + Om0*jnp.power((1.+z),3))

@jnp.vectorize
def d_L(z,H0,Om0):
    zs = jnp.linspace(0,z,100)
    integral = jnp.trapz(Ez_inv(zs,Om0),zs)
    return (1+z)*integral * (Clight/1e3)  / H0 #Mpc

def diff_comoving_volume(z,H0,Om0):
    dL = d_L(z,H0,Om0) #Mpc
    Ez_i = Ez_inv(z,Om0)
    D_H = (Clight/1e3)  / H0 #Mpc
    
    return 1.0e-9 * (4.*jnp.pi) * jnp.power(dL,2) * D_H * Ez_i / jnp.power(1.+z,2.)

def z_at_dl(dl,H0,Om0,zmin=1e-3,zmax=100):
    #dl in Mpc
    zs = jnp.logspace(jnp.log10(zmin),jnp.log10(zmax),1000)
    return jnp.interp(dl, d_L(zs,H0,Om0),zs, left=zmin, right=zmax, period=None)

def detector_to_source_frame(m1z,m2z,dL,H0,Om0,zmin=1e-3,zmax=100):
    z = z_at_dl(dL,H0,Om0,zmin,zmax)
    m1 = m1z / (1. + z)
    m2 = m2z / (1. + z)
    return m1, m2, z

"""Approximate luminosity distance"""
#https://arxiv.org/pdf/1111.6396.pdf
def Phi(x):
    num = 1 + 1.320*x + 0.4415* jnp.power(x,2) + 0.02656*jnp.power(x,3)
    den = 1 + 1.392*x + 0.5121* jnp.power(x,2) + 0.03944*jnp.power(x,3)
    return num/den

def xx(z,Om0):
    return (1.0-Om0)/Om0/jnp.power(1.0+z,3)

def dL_approx(z,H0,Om0):
    D_H = (Clight/1.0e3)  / H0 #Mpc
    
    return 2.*D_H * (1.+z) * (Phi(xx(0.,Om0)) - Phi(xx(z,Om0))/jnp.sqrt(1.+z))/jnp.sqrt(Om0)

def z_at_dl_approx(dl,H0,Om0,zmin=1e-3,zmax=100):
    #dl in Mpc
    zs = jnp.logspace(jnp.log10(zmin),jnp.log10(zmax),1000)
    return jnp.interp(dl, dL_approx(zs,H0,Om0),zs, left=zmin, right=zmax, period=None)

def detector_to_source_frame_approx(m1z,m2z,dL,H0,Om0,zmin=1e-3,zmax=100):
    z = z_at_dl_approx(dL,H0,Om0,zmin,zmax)
    m1 = m1z / (1. + z)
    m2 = m2z / (1. + z)
    return m1, m2, z

def diff_comoving_volume_approx(z,H0,Om0):
    dL = dL_approx(z,H0,Om0) #Mpc
    Ez_i = Ez_inv(z,Om0)
    D_H = (Clight/1e3)  / H0 #Mpc
    
    return 1.0e-9 * (4.*jnp.pi) * jnp.power(dL,2) * D_H * Ez_i / jnp.power(1.+z,2.)

def dLdH_approx(z,Om0):    
    return 2. * (1.+z) * (Phi(xx(0.,Om0)) - Phi(xx(z,Om0))/jnp.sqrt(1.+z))/jnp.sqrt(Om0)

def z_at_dldH_approx(dl,Om0,zmin=1e-3,zmax=100):
    #dldH = H0*d_L/c is dimensionless
    zs = jnp.logspace(jnp.log10(zmin),jnp.log10(zmax),1000)
    return jnp.interp(dl, dLdH_approx(zs,Om0),zs, left=zmin, right=zmax, period=None)

def detector_to_source_frame_approx_dLdH(m1z,m2z,dLdH,Om0,zmin=1e-3,zmax=100):
    z = z_at_dldH_approx(dLdH,Om0,zmin,zmax)
    m1 = m1z / (1. + z)
    m2 = m2z / (1. + z)
    return m1, m2, z