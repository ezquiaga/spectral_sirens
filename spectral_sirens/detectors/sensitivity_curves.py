import numpy as np
from scipy.interpolate import interp1d

import os
sensitivity_curves_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'sensitivity_curves')
pw_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'pw_network')

def detector_psd(detector_name):
    file_detector, asd = get_filename(detector_name)
    file_detector = os.path.join(sensitivity_curves_path,file_detector)
    data = np.genfromtxt(file_detector)
    f = data[:,0]
    h = data[:,1]
    sn_int = interp1d(f,h**(1. + asd),bounds_error=False,fill_value=1.0)
    minf = min(f)
    maxf = max(f)
    return sn_int, minf, maxf
    
"""
Files for sensitivity curves
"""

detectors_list = ['CE-40', 'CE-20', 'ET-10-XYL', 'A#', 'A+', 'V+', 'K+', 'Voyager', 'ET']

def get_filename(tec):
    if   tec == 'CE-40':
        filename = 'cosmic_explorer_40km.txt'
        asd = 1
    elif tec == 'CE-20':
        filename = 'cosmic_explorer_20km.txt'
        asd = 1
    # https://apps.et-gw.eu/tds/?content=3&r=18213 --> 1st (frequencies) and 4th (xylophone PSD) columns of ET10kmcolumns.txt
    elif tec == 'ET-10-XYL':
        filename = 'et_10km_xylophone.txt'
        asd = 0
    # https://dcc.ligo.org/LIGO-T2300041-v1/public
    elif tec == 'A#':
        filename = 'a_sharp.txt'
        asd = 1
    # curves used in the trade study for the Cosmic Explorer Horizon Study, see https://dcc.cosmicexplorer.org/CE-T2000007/public
    elif tec == 'A+':
        filename = 'a_plus.txt'
        asd = 1
    elif tec == 'V+':
        filename = 'advirgo_plus.txt'
        asd = 1
    elif tec == 'K+':
        filename = 'kagra_plus.txt'
        asd = 1
    elif tec == 'Voyager':
        filename = 'voyager.txt'
        asd = 1
    elif tec == 'ET':
        filename = 'et_d.txt'
        asd = 1
    else: raise ValueError(f'Specified detector sensitivity "{tec}" not found in {detectors_list}.')

    return filename, asd


def detector_name(detector):
    if detector == 'CE-40':
        name = 'Cosmic Explorer 40km'
    elif detector == 'CE-20':
        name = 'Cosmic Explorer 20km'
    elif detector == 'ET-10-XYL':
        name = 'Einstein Telescope 10km Xylophone'
    elif detector == 'A#':
        name = 'Advanced LIGO Sharp'
    elif detector == 'A+':
        name = 'Advanced LIGO Plus'
    elif detector == 'V+':
        name = 'Advanced Virgo Plus'
    elif detector == 'K+':
        name = 'KAGRA Plus'
    elif detector == 'Voyager':
        name = 'Voyager'
    elif detector == 'ET':
        name = 'Einstein Telescope'
    else: raise ValueError(f'Specified detector sensitivity "{detector}" not found in {detectors_list}.')
    return name

"""Antenna patter sensitivity detector for different networks"""
w_data, pw_data = np.genfromtxt(os.path.join(pw_path,'pw_single.txt'),unpack=True)
pw=interp1d(w_data, pw_data,bounds_error=False,fill_value=(1.0,0.0))

w_single, pw_single = np.genfromtxt(os.path.join(pw_path,'pw_single.txt'),unpack=True)
pw_single=interp1d(w_single, pw_single,bounds_error=False,fill_value=(1.0,0.0))
w_hl, pw_hl = np.genfromtxt(os.path.join(pw_path,'pw_hl.txt'),unpack=True)
pw_hl=interp1d(w_hl, pw_hl,bounds_error=False,fill_value=(1.0,0.0))
w_hlv, pw_hlv = np.genfromtxt(os.path.join(pw_path,'pw_hlv.txt'),unpack=True)
pw_hlv=interp1d(w_hlv, pw_hlv,bounds_error=False,fill_value=(1.0,0.0))
w_hlvji, pw_hlvji = np.genfromtxt(os.path.join(pw_path,'pw_hlvji.txt'),unpack=True)
pw_hlvji=interp1d(w_hlvji, pw_hlvji,bounds_error=False,fill_value=(1.0,0.0))