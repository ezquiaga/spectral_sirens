import numpy as np
from scipy.integrate import trapz

from spectral_sirens.cosmology import gwcosmo
from spectral_sirens.utils import gwutils

def dNcbc_dz(z,pz,R0,H0,Om0,Tobs):
    #rate in yr^-1 Gpc^-3
    #Tobs in *detector* frame

    Rz = R0 * pz(z)
    Vol = gwcosmo.diff_comoving_volume_approx(z,H0,Om0)

    integrand = Tobs * Rz * Vol / (1.+z) 
    return integrand
vdNcbc_dz = np.vectorize(dNcbc_dz)

def d3Ndet_dzdm1dm2(z,mass_1,mass_2,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based):
    #rate in yr^-1 Gpc^-3
    #f in *detector* frame
    #M in *source* frame
    #Tobs in *detector* frame
    Mc = gwutils.mchirp(mass_1,mass_2) #source frame masses
    dL = gwcosmo.dL_approx(z,H0,Om0)
    fmin_gw = fmin_detect # gwutils.f_ini(Tobs,Mc*(1.+z))
    mass_1z = mass_1*(1.+z)
    mass_2z = mass_2*(1.+z)
    snr_opt = gwutils.vsnr_from_psd(mass_1z,mass_2z,dL,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based)
    pw = sc.pw_hl(snr_th/snr_opt)

    pm = norm_m1 * pm1(mass_1) * pm2(mass_2,mass_1)
    integrand = dNcbc_dz(z,pz,R0,H0,Om0,Tobs) * pm * pw 
    return integrand
vd3Ndet_dzdm1dm2 = np.vectorize(d3Ndet_dzdm1dm2)

def d2Ndet_dzdm1(z,mass_1,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,n_m2):
    # Mass in Msun at *source* frame
    mass_2s = np.linspace(mmin,mass_1,n_m2)
    dn_detec = d3Ndet_dzdm1dm2(z,mass_1,mass_2s,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based)
    return trapz(dn_detec,mass_2s)
vd2Ndet_dzdm1 = np.vectorize(d2Ndet_dzdm1)

def dNdet_dz(z,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,mmax,n_m1,n_m2):
    # Mass in Msun at *source* frame
    mass_1s = np.linspace(mmin,mmax,n_m1)
    dn_detec = vd2Ndet_dzdm1(z,mass_1s,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,n_m2)
    return trapz(dn_detec,mass_1s)
vdNdet_dz = np.vectorize(dNdet_dz)

def Ndet(pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,mmax,n_m1,n_m2,zmin,zmax,n_z):
    # Mass in Msun at *source* frame
    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn_detec = vdNdet_dz(zs,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,mmax,n_m1,n_m2)
    return trapz(dn_detec,zs)

def Ncbc(pz,R0,H0,Om0,Tobs,zmin,zmax,n_z):
    # Mass in Msun at *source* frame
    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn = dNcbc_dz(zs,pz,R0,H0,Om0,Tobs)
    return trapz(dn,zs)
