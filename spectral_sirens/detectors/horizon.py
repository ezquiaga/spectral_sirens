import numpy as np
from spectral_sirens.utils import gwutils

"""Astropy"""
import astropy.units as astrou
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

"""Luminosity distance and redshift"""

def z_dl(dl):
    if dl<=1.0e-2:
        zdl = 0.
    elif dl<=1.0e12:
        zdl = z_at_value(cosmo.luminosity_distance,dl*astrou.kpc,zmax = 1.0e5)
    else:
        zdl = z_at_value(cosmo.luminosity_distance,1e12*astrou.kpc,zmax = 1.0e5)
    return zdl
vz_dl = np.vectorize(z_dl)

def z_dls(dl): #dl in kpc
    if dl < 1:
        zs = z_at_value(cosmo.luminosity_distance,1.*astrou.kpc,zmax = 1.0e5)
    elif dl < 1e11:
        zs = z_at_value(cosmo.luminosity_distance,dl*astrou.kpc,zmax = 1.0e5)
    else:
        zs = z_at_value(cosmo.luminosity_distance,1e11*astrou.kpc,zmax = 1.0e5)
    return zs
vz_dls = np.vectorize(z_dls)

"""Horizon distance"""
def dhor(mass_1z,mass_2z,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based,snr_th):
    # Mass in Msun at *detector* frame
    #f in *detector* frame
    #Tobs in *detector* frame
    dL = 1.0e-3 # Mpc
    return gwutils.vsnr_from_psd(mass_1z,mass_2z,dL,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based) / snr_th #in KPC


""" Maximum Horizon Distance """
def dhor_max(mass_1z,mass_2z,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based,snr_th):
    # Mass in Msun at *detector* frame
    dls = np.zeros(len(mass_1z))
    d = 1.0e-3 # Mpc
    for i in range(0,len(mass_1z)):
        fmax = gwutils.f_PhenomA(3,mass_1z[i],mass_2z[i])

        dli = np.zeros(len(fmin_gw))
        for j in range(0,len(fmin_gw)):
            if fmin_gw[j] < fmax:
                dli[j] = dhor(mass_1z[i],mass_2z[i],fmin_gw[j],Tobs,detectorSn,fmin_detect,fmax_detect,based,snr_th) #in KPC
                if dli[j] < 1.:
                    dli[j] = 1.
                elif dli[j] > 1.0e11: #Avoids divergences, should improve this!
                    dli[j] = 1.
            else:
                dli[j] = 1.
        dls[i] = max(dli)
    return dls
vdhor_max = np.vectorize(dhor_max)

def zhor_max(mass_1z,mass_2z,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based,snr_th):
    return vz_dls(dhor_max(mass_1z,mass_2z,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based,snr_th))
