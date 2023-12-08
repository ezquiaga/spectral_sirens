#Fiducial universe
#-----------------

#Cosmology
#---------
H0_fid = 67.66 # cosmo.H(0).value #cosmology is Planck'18
Om0_fid = 0.30966 # cosmo.Om(0) #cosmology is Planck'18

#Merger rate
#-----------
# It follows the star formation rate from Madau & Dickinson (2014) (https://arxiv.org/abs/1403.0007)
r0_fid = 30.
alpha_z_fid = 2.7
zp_fid = 1.9
beta_fid = 5.6 - alpha_z_fid

#Mass distribution
#-----------------
#Primary mass
mmin_pl_fid = 5. #Minimum mass for the power-law
mmax_pl_fid = 150. #Maximum mass for the power-law
mMin_filter_fid = 8.75 #Minimum mass for the low-mass filter
mMax_filter_fid = 87. #Maximum mass for the high-mass filter
dmMin_filter_fid = 1. 
dmMax_filter_fid = 2.
alpha_fid = -3.4
sig_m1_fid = 3.6
mu_m1_fid = 34.
f_peak_fid = 0.04

#Mass ratio
bq_fid = 1.1

Tobs_fid = 1. #yr