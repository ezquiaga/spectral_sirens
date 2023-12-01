import numpy as np
from scipy.interpolate import interp1d

""" Inverse Sampling """
def inverse_transf_sampling(cum_values,variable,n_samples):
    inv_cdf = interp1d(cum_values, variable)
    r = np.random.uniform(min(cum_values),max(cum_values),n_samples)
    return inv_cdf(r)