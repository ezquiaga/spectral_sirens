import numpy as np
from scipy.interpolate import interp1d

""" Inverse Sampling """
def inverse_transf_sampling(cum_values,variable,n_samples):
    inv_cdf = interp1d(cum_values, variable)
    r = np.random.uniform(min(cum_values),max(cum_values),n_samples)
    return inv_cdf(r)

""" Useful functions  """
def powerlaw(m,mMin,mMax,alpha):
    if alpha == -1.:
        norm = 1 / np.log(mMax/mMin)
    else:
        norm = (1. + alpha)/(mMax**(alpha+1.) - mMin**(alpha+1.))
    prob = np.power(m,alpha)
    prob *= norm
    prob *= (m <= mMax) & (m >= mMin)
    return prob

def gaussian(x,mu,sig):
    return np.exp(-(x-mu)**2/(2.*sig**2))/np.sqrt(2.*np.pi*sig**2)

def lowfilter(m,mMin,dmMin):
    low_filter = np.exp(-(m-mMin)**2/(2.*dmMin**2))
    low_filter = np.where(m<mMin,low_filter,1.)
    return low_filter

def highfilter(m,mMax,dmMax):
    high_filter = np.exp(-(m-mMax)**2/(2.*dmMax**2))
    high_filter = np.where(m>mMax,high_filter,1.)
    return high_filter

def Sfilter(m,mMin,deltaM):
    """
    Smoothed filter function

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm,deltaMM):
        return np.exp(deltaMM/mm + deltaMM/(mm-deltaMM))
    
    S_filter = 1./(f(m-mMin,deltaM) + 1.)
    S_filter = np.where(m<mMin+deltaM,S_filter,1.)
    S_filter = np.where(m>mMin,S_filter,0.)
    return S_filter