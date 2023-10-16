import numpyro
import jax.numpy as jnp
from constants import *

""" Mass distribution """

def powerlaw(m,mMin,mMax,alpha):
    norm = (1. + alpha)/(mMax**(alpha+1.) - mMin**(alpha+1.))
    return norm * (m**alpha)

def powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    tmp_min = 2.
    tmp_max = 150.
    dmMax = 2
    dmMin = 1
    
    # Define power-law and peak
    p_m1_pl = powerlaw(m1,tmp_min,tmp_max,alpha)
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*jnp.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter

def logpowerlaw(m,mMin,mMax,alpha):
    lognorm = jnp.log(1. + alpha) - jnp.log(mMax**(alpha+1.) - mMin**(alpha+1.))
    return lognorm + alpha * jnp.log(m)

def logpowerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    tmp_min = 2.
    tmp_max = 150.
    dmMax = 2
    dmMin = 1
    
    # Define power-law and peak
    p_m1_pl = powerlaw(m1,tmp_min,tmp_max,alpha)
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*jnp.pi*sig_m1**2)

    # Compute low- and high-mass filters
    loglow_filter = -(m1-mMin)**2/(2.*dmMin**2)
    loglow_filter = jnp.where(m1<mMin,loglow_filter,0.)
    loghigh_filter = -(m1-mMax)**2/(2.*dmMax**2)
    loghigh_filter = jnp.where(m1>mMax,loghigh_filter,0.)

    # Apply filters to combined power-law and peak
    return jnp.log(f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl) + loghigh_filter + loglow_filter

def powerlaw_smooth(m,mMin,mMax,alpha):
    tmp_min = 2.
    tmp_max = 150.
    dmMax = 1
    dmMin = 1
    
    pm_pl = powerlaw(m,tmp_min,tmp_max,alpha)
    
    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m>mMax,high_filter,1.)
    
    return pm_pl*low_filter*high_filter

""" Redshift distribution """
def rconst(z):
    return 1.
def sfr(z):
    return 0.015*(1.+z)**2.7/(1.+((1.+z)/2.9)**5.6) #msun per yr per Mpc^3
def rsfr(z):
    return sfr(z)/sfr(0.)

def rate_z(z,zp,alpha,beta):
    c0 = 1. + (1. + zp)**(-alpha-beta)
    num = (1.+z)**alpha
    den = 1. + ((1.+z)/(1.+zp))**(alpha+beta)
    return c0 * num / den


""" Toy models """

def two_box(x,edge_1,width_1,edge_2,width_2,filter):
    def box_smooth(x,edge,width,filter):
        low_edge = edge
        high_edge = edge + width

        low_filter = jnp.exp(-(x-low_edge)**2/(2.*filter**2))
        low_filter = jnp.where(x<low_edge,low_filter,1.)
        high_filter = jnp.exp(-(x-high_edge)**2/(2.*filter**2))
        high_filter = jnp.where(x>high_edge,high_filter,1.)

        return low_filter*high_filter*1./width
    
    return box_smooth(x,edge_1,width_1,filter) + box_smooth(x,edge_2,width_2,filter)