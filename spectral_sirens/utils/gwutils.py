import numpy as np
from scipy.interpolate import interp1d
import scipy.stats as stats
from .constants import *
from ..detectors.sensitivity_curves import detector_psd

"""Frequency of the GW"""
def dtdf(fi,M): 
    #fi in Hz
    #M in Msun
    tM = M*TSUN
    return 5.*(tM**(-5./3.))*(fi**(-11./3.))/(96.*(np.pi**(8./3))) # in s

def t_merge(fi,M): 
    #fi in Hz
    #M in Msun in *detector* frame
    tM = M*TSUN
    return 5.*tM*((1./(8.*np.pi*tM*fi))**(8./3.))/YEAR # in yr

def f_ini(tmerge,M): 
    #tmerge in yr
    #M in Msun in *detector* frame
    tM = M*TSUN
    return (1./(8.*np.pi*tM))*(5.*tM/(tmerge*YEAR))**(3./8.) #Hz

def mchirp(m1,m2):
    return np.power(m1*m2,3./5.)/np.power(m1+m2,1./5.)

def eta(m1,m2):
    return m1*m2/((m1+m2)**2.)

def f_PhenomA(indx,m1,m2):
    a = np.array([2.9740e-1, 5.9411e-1, 5.0801e-1, 8.4845e-1])
    b = np.array([4.4810e-2, 8.9794e-2, 7.7515e-2, 1.2848e-1])
    c = np.array([9.5560e-2, 1.9111e-1, 2.2369e-2, 2.7299e-1])
    tM = (m1 + m2)*TSUN 
    eta = m1*m2/((m1+m2)**2.)
    num = a[indx]*eta**2 + b[indx]*eta + c[indx]
    return num/(np.pi*tM)

def f_fin(fi,m1,m2,T,*args): 
    #fi in Hz
    #M in Msun in *detector* frame
    #T of mission in yr
    Mc = mchirp(m1,m2)  
    if T < t_merge(fi,Mc):
        tM = Mc*TSUN 
        ffin = (1./(8.*np.pi*tM))*(-(T*YEAR)/(5.*tM) + (1./(8.*np.pi*tM*fi))**(8./3.))**(-3./8.)
    else:
        ffin = f_PhenomA(3,m1,m2)
        for arg in args:
            if arg == 'space':
                ffin = min(ffin,2.)
    return ffin # Hz
vf_fin = np.vectorize(f_fin)

"""Amplitude of the GW"""
from pycbc.waveform import get_fd_waveform

def aS(mass1,mass2,DL,fmin,Tobs,*args):
    #Masses in Msun at *detector* frame
    #Distance in MPC
    m1z = mass1
    m2z = mass2
    fmax = vf_fin(fmin,m1z,m2z,Tobs,*args)
    df = (fmax-fmin)/1000.
    if fmin > fmax:
        h_out = np.array([0., 0.])
        f_out = np.array([fmin, fmin*1.1])
    else:
        if df < 1.0e-6:
            df = 1.0e-6
        if (fmin + df) > fmax:
            fmax = fmin + 10.*df
        hp, hx = get_fd_waveform(approximant = 'IMRPhenomD',
                            mass1 = m1z,
                            mass2 = m2z,
                            spin1z = 0.,
                            spin2z = 0.,
                            distance = DL,
                            #inclination = 0*np.pi/1.9,
                            #long_asc_nodes = 0*np.pi/3.,
                            f_lower = fmin, 
                            f_final = fmax, 
                            delta_f = df)
        fs = hp.sample_frequencies  
        if fmin + 10.*df == fmax: # when the frequency does not vary
            rang = (fs >= fmin)
            hh = abs(hp)[rang]
            h_out = np.array([hh[0], 0.])
            f_out = np.array([fmin, vf_fin(fmin,m1z,m2z,Tobs,*args)])
        else:
            rang = (fs>=fmin) & (fs < fmax)
            h_out = abs(hp)[rang]
            f_out = np.linspace(min(fs[rang]),max(fs[rang]),len(fs[rang]))
            if len(h_out) < 2:
                h_out = np.array([abs(hp)[rang],abs(hp)[rang]])
                f_out = np.array([fmin, vf_fin(fmin,m1z,m2z,Tobs,*args)])
    return h_out, f_out
vaS = np.vectorize(aS)

def hA(f,M,d):
    # f in Hz
    # Mass in Msun at *detector* frame
    # d=luminosity distance in Mpc
    Mz = M
    num = ((Mz*TSUN)**(5./3.)) * ((np.pi*f)**(2./3.))
    den = d * MPC / Clight
    return 4. * num / den

"""Computing SNR"""
def snr_from_psd(mass1,mass2,DL,fmin,Tobs,detectorSn, fmin_detect, fmax_detect,*args):
    """
    Computes the SNR of a GW signal given the PSD of the detector.
    """
    # Mass in Msun at *detector* frame
    #f in *detector* frame
    #Tobs in *detector* frame
    # d=luminosity distance in Mpc
    Mc = mchirp(mass1,mass2)
    fmax = vf_fin(fmin,mass1,mass2,Tobs,*args)
    fmin = max(fmin,fmin_detect)
    fmax = min(fmax,fmax_detect)
    if fmin > fmax:
        snr2 = 1.0e-10
    elif fmax <fmin:
        snr2 = 1.0e-10
    else:
        logf = np.log(fmax)-np.log(fmin)
        if logf < 0.1 :
            if fmin < 0.8*f_PhenomA(0,mass1,mass2):
                snr2 = (Tobs*YEAR) * (hA(fmin,Mc,DL)**2)/detectorSn(fmin)
            else:
                h = aS(mass1,mass2,DL,fmin,Tobs,*args)[0]
                df = fmax-fmin
                if np.isnan(detectorSn(fmin))==False:
                    snr2 = 4.*(h[0]**2.)*df/detectorSn(fmin)
                else:
                    snr2 = 1.0e-10
        else: 
            h = aS(mass1,mass2,DL,fmin,Tobs,*args)[0]
            fs = aS(mass1,mass2,DL,fmin,Tobs,*args)[1]
            df = fs[1]-fs[0]
            rangf = (fs<fmax)
            integrand = (h[rangf]**2.)/detectorSn(fs[rangf])
            rang = np.isnan(detectorSn(fs[rangf]))==False 
            snr2 = 4.*np.sum(integrand[rang]*df)#np.sum(4.*integrand[rang]*df)
    return np.sqrt(snr2)
vsnr_from_psd = np.vectorize(snr_from_psd)

def snr(mass1,mass2,DL,fmin,Tobs,detector,*args):
    """
    Computes the SNR of a GW signal for a given detector.
    """
    # Mass in Msun at *detector* frame
    #f in *detector* frame
    #Tobs in *detector* frame
    # d=luminosity distance in Mpc
    detectorSn, fmin_detect, fmax_detect = detector_psd(detector)
    return snr_from_psd(mass1,mass2,DL,fmin,Tobs,detectorSn, fmin_detect, fmax_detect,*args)
vsnr = np.vectorize(snr)

def observed_snr(snr):
    lower_snr, upper_snr = 0.0, snr*10.
    mu_snr, sigma_snr = snr, 1.
    return stats.truncnorm.rvs((lower_snr - mu_snr) / sigma_snr, (upper_snr - mu_snr) / sigma_snr, loc=mu_snr, scale=sigma_snr,size=1)
observed_snr = np.vectorize(observed_snr)

"""Approximate errors in posteriors"""
def error_pe_detector(detector):
    sigma_Mc, sigma_eta, sigma_w = [3.0e-2,5.0e-3,5.0e-2]
    if any(detector == x for x in np.array(['O1','O2','O3'])):
        sigma_Mc, sigma_eta, sigma_w = [8.0e-2,2.2e-2,2.1e-1]
    elif any(detector == x for x in np.array(['O4'    ])):
        sigma_Mc, sigma_eta, sigma_w = [8.0e-2,1.0e-2,8.0e-2]
    elif any(detector == x for x in np.array(['O5','A+'])):
        sigma_Mc, sigma_eta, sigma_w = [3.0e-2,5.0e-3,5.0e-2]
    elif any(detector == x for x in np.array(['Voyager','A#'])):
        sigma_Mc, sigma_eta, sigma_w = [1.0e-2,2.0e-3,5.0e-2]
    elif any(detector == x for x in np.array(['ET','ET-10-XYL','CE-20','CE-40'])):
        sigma_Mc, sigma_eta, sigma_w = [5.0e-3,7.0e-4,2.0e-2]
    return sigma_Mc,sigma_eta,sigma_w

"""Observed posteriors"""
def observed_posteriors(n_samples,m1z,m2z,dL,snr_opt,snr_true,snr_obs,snr_th,fmin,Tobs,detector,*args):
    #clevel = 1.65 #90% CL
    sigma_Mc,sigma_eta,sigma_w = error_pe_detector(detector)
    
    Mz = mchirp(m1z,m2z) #source frame masses
    etas = eta(m1z,m2z)
    #Mz
    logMz_obs = np.random.normal(loc=np.log(Mz),scale=sigma_Mc* snr_th/snr_obs,size=n_samples)
    Mz_obs = np.exp(logMz_obs)
    #eta
    lower_etas, upper_etas = 0.0, 0.25
    mu_etas, sigma_etas = etas, sigma_eta* snr_th/snr_obs
    eta_obs = stats.truncnorm.rvs((lower_etas - mu_etas) / sigma_etas, (upper_etas - mu_etas) / sigma_etas, loc=mu_etas, scale=sigma_etas,size=n_samples)
    #w
    w = snr_true/snr_opt
    lower_ws, upper_ws = 0.0, 1.0
    mu_ws, sigma_ws = w, sigma_w* snr_th/snr_obs
    w_obs = stats.truncnorm.rvs((lower_ws - mu_ws) / sigma_ws, (upper_ws - mu_ws) / sigma_ws, loc=mu_ws, scale=sigma_ws, size=n_samples)
    
    M_obs = Mz_obs / eta_obs**(3./5.)
    m1z_obs = (M_obs + np.sqrt(M_obs**2 - 4.* eta_obs * M_obs**2))/2.
    m2z_obs = (M_obs - np.sqrt(M_obs**2 - 4.* eta_obs * M_obs**2))/2.
    
    snr_opt_1Mpc = vsnr(m1z_obs,m2z_obs,dL,fmin,Tobs,detector,*args)
    dL_obs = dL*snr_opt_1Mpc * w_obs / snr_obs #Mpc
    
    pdraw = np.abs(w_obs * snr_opt_1Mpc * (m1z_obs - m2z_obs) * np.power(eta(m1z_obs,m2z_obs),3./5)/np.power(m1z_obs + m2z_obs,2) / np.power(dL_obs,2))

    return m1z_obs, m2z_obs, dL_obs, pdraw
vobserved_posteriors = np.vectorize(observed_posteriors)