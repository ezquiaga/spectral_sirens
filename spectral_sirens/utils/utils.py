import numpy as np
from scipy.interpolate import interp1d

xp = np

""" Inverse Sampling """
def inverse_transf_sampling(cum_values,variable,n_samples):
    inv_cdf = interp1d(cum_values, variable)
    r = xp.random.uniform(min(cum_values),max(cum_values),n_samples)
    return inv_cdf(r)

""" Useful functions  """
def powerlaw(m,mMin,mMax,alpha):
    if alpha == -1.:
        norm = 1 / xp.log(mMax/mMin)
    else:
        norm = (1. + alpha)/(mMax**(alpha+1.) - mMin**(alpha+1.))
    prob = xp.power(m,alpha)
    prob *= norm
    prob *= (m <= mMax) & (m >= mMin)
    return prob

def gaussian(x,mu,sig):
    return xp.exp(-(x-mu)**2/(2.*sig**2))/xp.sqrt(2.*xp.pi*sig**2)

def lowfilter(m,mMin,dmMin):
    low_filter = xp.exp(-(m-mMin)**2/(2.*dmMin**2))
    low_filter = xp.where(m<mMin,low_filter,1.)
    return low_filter

def highfilter(m,mMax,dmMax):
    high_filter = xp.exp(-(m-mMax)**2/(2.*dmMax**2))
    high_filter = xp.where(m>mMax,high_filter,1.)
    return high_filter

def Sfilter(m,mMin,deltaM):
    """
    Smoothed filter function

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm,deltaMM):
        return xp.exp(deltaMM/mm + deltaMM/(mm-deltaMM))
    
    S_filter = 1./(f(m-mMin,deltaM) + 1.)
    S_filter = xp.where(m<mMin+deltaM,S_filter,1.)
    S_filter = xp.where(m>mMin,S_filter,0.)
    return S_filter