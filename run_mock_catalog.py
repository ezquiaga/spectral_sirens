#IMPORT
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz, trapz
import scipy.stats as stats
import time


#PYTHON MODULES
from constants import *
import gwcosmo
import gwpop
import gwutils
import utils
import sensitivity_curves as sc

#Output directory
dir_out = 'data_mock_catalogues/'
#Outputs
PATH = "/groups/astro/ezquiaga/spectral_sirens/"

#Fiducial universe
from fiducial_universe import *
model_name = 'powerlaw_peak_alpha_%s_sig_%s_mu_%s_fpeak_%s_mmin_%s_mmax_%s'%(alpha_fid, sig_m1_fid, mu_m1_fid, f_peak_fid, mmin_fid, mmax_fid)

#Mock catalogue parameters
n_detections = 999
n_samples = 20
n_sources = n_detections*10
#random_seed = np.random.seed(2)

#Detector configuration
fmin = 10.
Tobs = 1.
detector = 'O5'
det_Sc = sc.O5
based = 'ground'
snr_th = 8.0

#We define the source populations in source masses and redshifts
# In this case we choose:
# - Refshift: SFR like function
# - Primary mass: smooth power-law+peak model
# - Secondary mass: uniform in mass ratio

def mock_source_parameters(n_sources,H0,Om0,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,zp,alpha_z,beta):
    zs = np.linspace(0.01,10,1000)
    cdf_z = cumtrapz(gwcosmo.diff_comoving_volume_approx(zs,H0,Om0)*gwpop.rate_z(zs,zp,alpha_z,beta)/(1+zs),zs,initial=0)
    cdf_z /= cdf_z[-1]
    masses = np.linspace(tmp_min,tmp_max,1000)
    cdf_m1 = cumtrapz(gwpop.powerlaw_peak(masses,mMin,mMax,alpha,sig_m1,mu_m1,f_peak),masses,initial=0)
    cdf_m1 /= cdf_m1[-1]

    z_mock = utils.inverse_transf_sampling(cdf_z,zs,n_sources)
    m1_mock = utils.inverse_transf_sampling(cdf_m1,masses,n_sources)
    q_mock = np.random.uniform(0,1,n_sources)
    m2_mock = m1_mock * q_mock
    dL_mock = gwcosmo.dL_approx(z_mock,H0,Om0)
    m1z_mock = (1 + z_mock) * m1_mock
    m2z_mock = (1 + z_mock) * m2_mock
    return m1z_mock, m2z_mock, dL_mock

def mock_population(n_sources,n_detections,n_samples,H0,Om0,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,zp,alpha_z,beta,snr_th,fmin,Tobs,detector,*args):
    #n_sources : number of sources to run code
    #n_samples : number of posterior samples per detection
    
    #Mock source paramters
    ww = np.linspace(0.0,1.0,1000)
    cdf_ww = 1.0-gwutils.pw_hl(ww)
    m1z_mock_pop, m2z_mock_pop, dL_mock_pop = mock_source_parameters(n_sources,H0,Om0,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,zp,alpha_z,beta)
    
    #SNR calcultion: optimal SNR -> true SNR -> observed SNR
    snr_opt_mock_pop = gwutils.vsnr(m1z_mock_pop,m2z_mock_pop,dL_mock_pop,fmin,Tobs,det_Sc,based)
    w_mock_pop = utils.inverse_transf_sampling(cdf_ww,ww,n_sources) #random draw
    snr_true_mock_pop = snr_opt_mock_pop*w_mock_pop
    snr_obs_mock_pop = gwutils.observed_snr(snr_true_mock_pop)

    #Detected population
    detected = snr_obs_mock_pop > snr_th
    snr_opt_mock = snr_opt_mock_pop[detected]
    snr_true_mock = snr_true_mock_pop[detected]
    snr_obs_mock = snr_obs_mock_pop[detected]
    m1z_mock = m1z_mock_pop[detected]
    m2z_mock = m2z_mock_pop[detected]
    dL_mock = dL_mock_pop[detected]
    
    snr_opt_mock = snr_opt_mock_pop[detected]
    snr_true_mock = snr_true_mock_pop[detected]
    snr_obs_mock = snr_obs_mock_pop[detected]
    m1z_mock = m1z_mock_pop[detected]
    m2z_mock = m2z_mock_pop[detected]
    dL_mock = dL_mock_pop[detected]
    
    while np.size(snr_opt_mock) < n_detections:    
        m1z_mock_pop, m2z_mock_pop, dL_mock_pop = mock_source_parameters(n_sources,H0,Om0,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,zp,alpha_z,beta)
        snr_opt_mock_pop = gwutils.vsnr(m1z_mock_pop,m2z_mock_pop,dL_mock_pop,fmin,Tobs,det_Sc,based)
        w_mock_pop = utils.inverse_transf_sampling(cdf_ww,ww,n_sources) #random draw
        snr_true_mock_pop = snr_opt_mock_pop*w_mock_pop
        snr_obs_mock_pop = gwutils.observed_snr(snr_true_mock_pop)
    
        detected = snr_obs_mock_pop > snr_th
        snr_opt_mock_add = snr_opt_mock_pop[detected]
        snr_true_mock_add = snr_true_mock_pop[detected]
        snr_obs_mock_add = snr_obs_mock_pop[detected]
        m1z_mock_add = m1z_mock_pop[detected]
        m2z_mock_add = m2z_mock_pop[detected]
        dL_mock_add = dL_mock_pop[detected]
    
        snr_opt_mock = np.append(snr_opt_mock,snr_opt_mock_add)
        snr_true_mock = np.append(snr_true_mock,snr_true_mock_add)
        snr_obs_mock = np.append(snr_obs_mock,snr_obs_mock_add)
        m1z_mock = np.append(m1z_mock,m1z_mock_add)
        m2z_mock = np.append(m2z_mock,m2z_mock_add)
        dL_mock = np.append(dL_mock,dL_mock_add)
    
    snr_opt_mock_det = snr_opt_mock[0:n_detections]
    snr_true_mock_det = snr_true_mock[0:n_detections]
    snr_obs_mock_det = snr_obs_mock[0:n_detections]
    m1z_mock_det = m1z_mock[0:n_detections] 
    m2z_mock_det = m2z_mock[0:n_detections]
    dL_mock_det = dL_mock[0:n_detections]
    
    m1z_mock_samples = np.zeros((n_detections,n_samples))
    m2z_mock_samples = np.zeros((n_detections,n_samples))
    dL_mock_samples = np.zeros((n_detections,n_samples))
    pdraw_mock_samples = np.zeros((n_detections,n_samples))
    for i in range(n_detections):
        m1z_mock_samples[i,:], m2z_mock_samples[i,:], dL_mock_samples[i,:], pdraw_mock_samples[i,:] = gwutils.observed_posteriors(n_samples,
                                                          m1z_mock_det[i],
                                                          m2z_mock_det[i],
                                                          dL_mock_det[i],
                                                          snr_opt_mock_det[i],
                                                          snr_true_mock_det[i],
                                                          snr_obs_mock_det[i],
                                                          snr_th,
                                                          fmin,
                                                          Tobs,
                                                          detector,
                                                          *args)
        
    return m1z_mock_samples, m2z_mock_samples, dL_mock_samples, pdraw_mock_samples

#Run mock population
m1z_mock_samples = np.zeros((n_detections,n_samples))
m2z_mock_samples = np.zeros((n_detections,n_samples))
dL_mock_samples = np.zeros((n_detections,n_samples))
pdraw_mock_samples = np.zeros((n_detections,n_samples))

starttime = time.time()
m1z_mock_samples,m2z_mock_samples,dL_mock_samples,pdraw_mock_samples = mock_population(n_sources,
                                        n_detections,
                                        n_samples,
                                        H0_fid,
                                        Om0_fid,
                                        mmin_fid,
                                        mmax_fid,
                                        alpha_fid,
                                        sig_m1_fid,
                                        mu_m1_fid,
                                        f_peak_fid,
                                        zp_fid,
                                        alpha_z_fid,
                                        beta_fid,
                                        snr_th,
                                        fmin,
                                        Tobs,
                                        det_Sc,
                                        based)
print('Time taken = {} seconds'.format(time.time() - starttime))

#Save data
###########
np.save(f'{PATH}'+dir_out+'m1z_'+detector+'_Ndet_%s_Nsamples_%s_'% (n_detections,n_samples)+model_name,m1z_mock_samples)
np.save(f'{PATH}'+dir_out+'m2z_'+detector+'_Ndet_%s_Nsamples_%s_'% (n_detections,n_samples)+model_name,m2z_mock_samples)
np.save(f'{PATH}'+dir_out+'dL_'+detector+'_Ndet_%s_Nsamples_%s_'% (n_detections,n_samples) +model_name,dL_mock_samples)
np.save(f'{PATH}'+dir_out+'pdraw_'+detector+'_Ndet_%s_Nsamples_%s_'% (n_detections,n_samples)+model_name,pdraw_mock_samples)