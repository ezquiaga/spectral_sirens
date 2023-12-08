import jax
import jax.numpy as jnp

xp = jnp

def logdiffexp(x, y): 
        return x + xp.log1p(xp.exp(y-x))

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

""" Filter functions """
def sigmoid(x,edge,width):
    #1./(1.+xp.exp(-(x-edge)/width))
    exponent = (x-edge)/width
    return jax.nn.sigmoid(exponent)

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

""" Log filter functions """

def loglowfilter(m,mMin,dmMin):
    low_filter = -(m-mMin)**2/(2.*dmMin**2)
    low_filter = xp.where(m<mMin,low_filter,0.)
    return low_filter

def loghighfilter(m,mMax,dmMax):
    high_filter = -(m-mMax)**2/(2.*dmMax**2)
    high_filter = xp.where(m>mMax,high_filter,0.)
    return high_filter