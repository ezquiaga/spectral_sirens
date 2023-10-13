import jax.numpy as jnp
from constants import *

def mchirp(m1,m2):
    return jnp.power(m1*m2,3./5.)/jnp.power(m1+m2,1./5.)

def eta(m1,m2):
    return m1*m2/((m1+m2)**2.)