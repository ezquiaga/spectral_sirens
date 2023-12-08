import jax.numpy as jnp
from ..utils import jutils

xp = jnp

""" Mass distribution """
def powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    # Define power-law and peak
    p_m1_pl = jutils.powerlaw(m1,mMin,mMax,alpha)
    p_m1_peak = jutils.gaussian(m1,mu_m1,sig_m1)

    # Combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)

def powerlaw_peak_smooth(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter):
    """
    Smoothed power-law + peak distribution

    Parameters
    ----------
    m1 : array_like
        Component mass
    mMin : float
        Minimum component mass for power-law
    mMax : float
        Maximum component mass for power-law
    alpha : float
        Power-law index
    sig_m1 : float
        Width of Gaussian peak
    mu_m1 : float
        Mean of Gaussian peak
    f_peak : float
        Fraction of events in Gaussian peak
    mMin_filter : float
        Minimum component mass for low-mass filter
    mMax_filter : float
        Maximum component mass for high-mass filter
    dmMin_filter : float
        Width of low-mass filter
    dmMax_filter : float
        Width of high-mass filter

    Returns
    -------
    array_like
        Smoothed power-law + peak distribution
    """
    # Powerlaw_peak
    plp = powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak)

    # Compute low- and high-mass filters
    low_filter = jutils.lowfilter(m1,mMin_filter,dmMin_filter)
    high_filter = jutils.highfilter(m1,mMax_filter,dmMax_filter)

    # Apply filters to combined power-law and peak
    return plp*low_filter*high_filter

def powerlaw_peak_gwtc3(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,deltaM):
    """
    Smoothed power-law + peak distribution used in GWTC-3

    See Eq. B4 in https://arxiv.org/pdf/2111.03634.pdf

    Parameters
    ----------
    m1 : array_like
        Component mass
    mMin : float
        Minimum component mass
    mMax : float
        Maximum component mass
    alpha : float
        Power-law index
    sig_m1 : float
        Width of Gaussian peak
    mu_m1 : float
        Mean of Gaussian peak
    f_peak : float
        Fraction of events in Gaussian peak
    deltaM : float
        Width of smoothed filter function

    Returns
    -------
    array_like
        Smoothed power-law + peak distribution used in GWTC-3
    """
    # Powerlaw_peak
    plp = powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak)

    # Compute low- and high-mass filters
    low_filter = jutils.Sfilter(m1,mMin,deltaM)

    # Apply filters to combined power-law and peak
    return plp*low_filter

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

def box_smooth(x,edge,width,filt):
    low_edge = edge
    high_edge = edge + width

    low_filter = jutils.lowfilter(x,low_edge,filt)
    high_filter = jutils.highfilter(x,high_edge,filt)

    return low_filter*high_filter*1./width

def two_box(x,edge_1,width_1,edge_2,width_2,filt,switch):
    
    return xp.where(x < switch,box_smooth(x,edge_1,width_1,filt),box_smooth(x,edge_2,width_2,filt))

def box_sig(x,edge,width,filt):
    low_edge = edge
    high_edge = edge + width
    mid_point = edge + width/2.

    low_filter = jutils.sigmoid(x,low_edge-2*filt,filt)
    low_filter = xp.where(x<mid_point,low_filter,1.)
    high_filter = jutils.sigmoid(-x,-high_edge-2*filt,filt)
    high_filter = xp.where(x>mid_point,high_filter,1.)

    return low_filter*high_filter*1./width

def two_box_sig(x,edge_1,width_1,edge_2,width_2,filt,switch):
    
    return xp.where(x < switch,box_sig(x,edge_1,width_1,filt),box_sig(x,edge_2,width_2,filt))

def uniform_sigmoid(x,high_edge,width,filt):
    low_edge = high_edge - width
    mid_point = high_edge - width/2.

    low_filter = jutils.sigmoid(x,low_edge-2*filt,filt)
    low_filter = xp.where(x<mid_point,low_filter,1.)
    high_filter = jutils.sigmoid(-x,-high_edge-2*filt,filt)
    high_filter = xp.where(x>mid_point,high_filter,1.)

    return low_filter*high_filter*1./width