import numpy as np
from scipy.integrate import cumtrapz
import scipy.stats as stats
import h5py


#PYTHON MODULES
from constants import *
import gwcosmo
import gwpop
import gwutils
import utils
import sensitivity_curves as sc

#Output directory
dir_out = 'data_injections/'

#Fiducial universe
from fiducial_universe import *

#Detector configuration
fmin = 10.
detector = 'O5'
det_Sc = sc.O5
based = 'ground'
snr_th = 8.

#Injection parameters
#- Uniform log dL
#- Power law for detector primary mass 
#- Uniforma mass ratio
params = 'm1z_m2z_dL'
alpha_inj, mmin_inj, mmax_inj = -.3, 1., 1000.

#Number of injections
n_detections = int(1e5)
n_sources = n_detections*50
#random_seed = np.random.seed(42)

##############
#Injections
##############

##Defining the injected distribution CDFs
m1zs = np.linspace(mmin_inj,mmax_inj,10000)
cdf_m1z = cumtrapz(gwpop.power_law(m1zs,alpha_inj, mmin_inj,mmax_inj),m1zs,initial=0.0)
cdf_m1z = cdf_m1z / cdf_m1z[-1]
#Geometric factor from orientations
ww = np.linspace(0.0,1.0,1000)
cdf_ww = 1.0-gwutils.pw_hl(ww)

##Computing injected events
#----
#Detector frame primary mass
mz_min,mz_max = mmin_inj,mmax_inj
m1z_mock_pop = utils.inverse_transf_sampling(cdf_m1z,m1zs,n_sources)
m2z_mock_pop = np.random.uniform(mz_min,m1z_mock_pop,n_sources)
#Luminosity distance
log10dL_min,log10dL_max = 1., np.log10(27000.0)
log10dL_mock_pop = np.random.uniform(log10dL_min,log10dL_max,n_sources)
dL_mock_pop = np.power(10.0,log10dL_mock_pop)

#Optimal SNR
snr_opt_mock_pop = gwutils.vsnr(m1z_mock_pop,m2z_mock_pop,dL_mock_pop,fmin,Tobs_fid,det_Sc,based)
#True SNR
w_mock_pop = utils.inverse_transf_sampling(cdf_ww,ww,n_sources) #random draw
snr_true_mock_pop = snr_opt_mock_pop*w_mock_pop
#Observed SNR
snr_obs_mock_pop = gwutils.observed_snr(snr_true_mock_pop)

##Computing p_draw
p_draw_m1z = gwpop.power_law(m1z_mock_pop,alpha_inj, mmin_inj,mmax_inj)
p_draw_m2z = 1./(m1z_mock_pop-mz_min)
p_draw_logdL = 1./(log10dL_max-log10dL_min)
jac_logdLdL = np.log10(np.exp(1.))/dL_mock_pop
p_draw_mock_pop = p_draw_m1z * p_draw_logdL * jac_logdLdL * p_draw_m2z

#Detected injections
m1z_inj = m1z_mock_pop[snr_obs_mock_pop>snr_th]
m2z_inj = m2z_mock_pop[snr_obs_mock_pop>snr_th]
dL_inj = dL_mock_pop[snr_obs_mock_pop>snr_th]
p_draw_inj = p_draw_mock_pop[snr_obs_mock_pop>snr_th]

Ndet = np.size(m1z_inj)
Ndraws = n_sources
print('Ndet = ',Ndet,', Ndraw = ',Ndraws)

# Saving the data
variables = ['m1z_inj','m2z_inj','dL_inj','p_draw_inj']
with h5py.File(dir_out+'injections_'+detector+'_'+params+'_Ndraws_%s_Ndet_%s.hdf5' % (Ndraws,Ndet), "w") as f:
    for var in variables:
        dset = f.create_dataset(var, data=eval(var))