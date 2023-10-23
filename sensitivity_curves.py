import numpy as np
from scipy.interpolate import interp1d


""" Ground based detectors """

dataAplus = np.genfromtxt('data_sensitivity_curves/aplus.txt')
fAplus=dataAplus[:,0]
hAplus=dataAplus[:,1]
sn_intAplus = interp1d(fAplus,hAplus**2.,bounds_error=False,fill_value=1.0)
minfAplus = min(fAplus)
maxfAplus = max(fAplus)
def Aplus():
    return sn_intAplus, minfAplus, maxfAplus

dataO1 = np.genfromtxt('data_sensitivity_curves/o1.txt')
fO1=dataO1[:,0]
hO1=dataO1[:,1]
sn_intO1 = interp1d(fO1,hO1**2.,bounds_error=False,fill_value=1.0)
minfO1 = min(fO1)
maxfO1 = max(fO1)
def O1():
    return sn_intO1, minfO1, maxfO1

dataO2 = np.genfromtxt('data_sensitivity_curves/o2.txt')
fO2=dataO2[:,0]
hO2=dataO2[:,1]
sn_intO2 = interp1d(fO2,hO2**2.,bounds_error=False,fill_value=1.0)
minfO2 = min(fO2)
maxfO2 = max(fO2)
def O2():
    return sn_intO2, minfO2, maxfO2

dataO3 = np.genfromtxt('data_sensitivity_curves/aligo_O3actual_L1.txt')
fO3=dataO3[:,0]
hO3=dataO3[:,1]
sn_intO3 = interp1d(fO3,hO3**2.,bounds_error=False,fill_value=1.0)
minfO3 = min(fO3)
maxfO3 = max(fO3)
def O3():
    return sn_intO3, minfO3, maxfO3

dataO4 = np.genfromtxt('data_sensitivity_curves/aligo_O4high.txt')
fO4=dataO4[:,0]
hO4=dataO4[:,1]
sn_intO4 = interp1d(fO4,hO4**2.,bounds_error=False,fill_value=1.0)
minfO4 = min(fO4)
maxfO4 = max(fO4)
def O4():
    return sn_intO4, minfO4, maxfO4

dataO5 = np.genfromtxt('data_sensitivity_curves/AplusDesign.txt')
fO5=dataO5[:,0]
hO5=dataO5[:,1]
sn_intO5 = interp1d(fO5,hO5**2.,bounds_error=False,fill_value=1.0)
minfO5 = min(fO5)
maxfO5 = max(fO5)
def O5():
    return sn_intO5, minfO5, maxfO5


datavoyager = np.genfromtxt('data_sensitivity_curves/voyager.txt')
fvoyager=datavoyager[:,0]
hvoyager=datavoyager[:,1]
sn_intvoyager = interp1d(fvoyager,hvoyager**2.,bounds_error=False,fill_value=1.0)
minfvoyager = min(fvoyager)
maxfvoyager = max(fvoyager)
def voyager():
    return sn_intvoyager, minfvoyager, maxfvoyager

dataET = np.genfromtxt('data_sensitivity_curves/et_d.txt')
fET=dataET[:,0]
hET=dataET[:,1]
sn_intET = interp1d(fET,hET**2.,bounds_error=False,fill_value=1.0)
minfET = min(fET)
maxfET = max(fET)
def ET():
    return sn_intET, minfET, maxfET

dataCE = np.genfromtxt('data_sensitivity_curves/ce.txt')
fCE=dataCE[:,0]
hCE=dataCE[:,1]
sn_intCE = interp1d(fCE,hCE**2.,bounds_error=False,fill_value=1.0)
minfCE = min(fCE)
maxfCE = max(fCE)
def CE():
    return sn_intCE, minfCE, maxfCE

""" Virgo """

dataVO3 = np.genfromtxt('data_sensitivity_curves/avirgo_O3actual.txt')
fVO3=dataVO3[:,0]
hVO3=dataVO3[:,1]
sn_intVO3 = interp1d(fVO3,hVO3**2.,bounds_error=False,fill_value=1.0)
minfVO3 = min(fVO3)
maxfVO3 = max(fVO3)
def VO3():
    return sn_intVO3, minfVO3, maxfVO3

dataVO4 = np.genfromtxt('data_sensitivity_curves/avirgo_O4high_NEW.txt')
fVO4=dataVO4[:,0]
hVO4=dataVO4[:,1]
sn_intVO4 = interp1d(fVO4,hVO4**2.,bounds_error=False,fill_value=1.)
minfVO4 = min(fVO4)
maxfVO4 = max(fVO4)
def VO4():
    return sn_intVO4, minfVO4, maxfVO4

dataVO5 = np.genfromtxt('data_sensitivity_curves/avirgo_O5high_NEW.txt')
fVO5=dataVO5[:,0]
hVO5=dataVO5[:,1]
sn_intVO5 = interp1d(fVO5,hVO5**2.,bounds_error=False,fill_value=1.)
minfVO5 = min(fVO5)
maxfVO5 = max(fVO5)
def VO5():
    return sn_intVO5, minfVO5, maxfVO5