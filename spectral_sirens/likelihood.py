import jax.numpy as jnp
from constants import *
import jgwcosmo

#Merger rate likelihood
def log_Rz(z,r0,zp,alpha,beta):
    logc0 = jnp.log1p((1. + zp)**(-alpha-beta))
    return jnp.log(r0) + logc0  + alpha*jnp.log1p(z) - jnp.log1p(jnp.power((1.+z)/(1.+zp),(alpha+beta)))

#Cosmological likelihood 
def log_cosmo_dL(z,dL,H0,Om0):
    Ez_i = jgwcosmo.Ez_inv(z,Om0)
    D_H = (Clight/1.0e3)  / H0 #Mpc
    
    logdiff_comoving_volume = jnp.log(1.0e-9) + jnp.log(4.0*jnp.pi) + 2.0*jnp.log(dL) +jnp.log(D_H) +jnp.log(Ez_i)-2*jnp.log1p(z)
    ddLdz = dL/(1.+z) + (1. + z)*D_H * Ez_i #Mpc 
    logJacobian_dL_z = - jnp.log(jnp.abs(ddLdz)) #Jac has absolute value 
    logJacobian_t_td = - jnp.log1p(z)
    return logdiff_comoving_volume + logJacobian_t_td + logJacobian_dL_z 

def log_cosmo(z,H0,Om0):
    dL = jgwcosmo.dL_approx(z,H0,Om0)#Mpc
    return log_cosmo_dL(z,dL,H0,Om0)

#Likelihood difference
def logdiffexp(x, y):
        return x + jnp.log1p(jnp.exp(y-x))