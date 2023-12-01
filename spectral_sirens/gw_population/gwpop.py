import numpy as np

""" Mass distribution """
def power_law(m1,alpha,mmin,mmax):
    if m1 < mmin:
        pdf = 0.
    elif m1 > mmax:
        pdf = 0.
    else:
        pdf = (alpha+1.)*(m1**alpha)/((mmax)**(alpha+1.)-mmin**(alpha+1.))
    return pdf
power_law = np.vectorize(power_law)

def powerlaw(m,mMin,mMax,alpha):
    norm = (1. + alpha)/(mMax**(alpha+1.) - mMin**(alpha+1.))
    return norm * (m**alpha)

tmp_min = 2.
tmp_max = 150.
dmMax = 2
dmMin = 1

def powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    # Define power-law and peak
    p_m1_pl = powerlaw(m1,tmp_min,tmp_max,alpha)
    p_m1_peak = np.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/np.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = np.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = np.where(m1<mMin,low_filter,1.)
    high_filter = np.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = np.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter


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

    low_filter = np.exp(-(x-low_edge)**2/(2.*filt**2))
    low_filter = np.where(x<low_edge,low_filter,1.)
    high_filter = np.exp(-(x-high_edge)**2/(2.*filt**2))
    high_filter = np.where(x>high_edge,high_filter,1.)

    return low_filter*high_filter*1./width

def two_box(x,edge_1,width_1,edge_2,width_2,filt,switch):
    
    return np.where(x < switch,box_smooth(x,edge_1,width_1,filt),box_smooth(x,edge_2,width_2,filt))

def sigmoid(x,edge,width):
    return 1./(1.+np.exp(-(x-edge)/width))

def box_sig(x,edge,width,filt):
    low_edge = edge
    high_edge = edge + width
    mid_point = edge + width/2.

    low_filter = sigmoid(x,low_edge-2*filt,filt)
    low_filter = np.where(x<mid_point,low_filter,1.)
    high_filter = sigmoid(-x,-high_edge-2*filt,filt)
    high_filter = np.where(x>mid_point,high_filter,1.)

    return low_filter*high_filter*1./width

def two_box_sig(x,edge_1,width_1,edge_2,width_2,filt,switch):
    
    return np.where(x < switch,box_sig(x,edge_1,width_1,filt),box_sig(x,edge_2,width_2,filt))

def gaussian(x,mu,sig):
    return np.exp(-(x-mu)**2/(2.*sig**2))/np.sqrt(2.*np.pi*sig**2)