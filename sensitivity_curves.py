import numpy as np
from scipy.interpolate import interp1d


""" Ground based detectors """

#READING DATA
#dataKAGRA = np.genfromtxt('data_sensitivity_curves/fig1_kagra_sensitivity.txt')
#fKAGRA=dataKAGRA[:,0]
#hKAGRA=dataKAGRA[:,5]

#dataVIRGO = np.genfromtxt('data_sensitivity_curves/fig1_adv_sensitivity.txt')
#fVIRGO=dataVIRGO[:,0]
#hVIRGO=dataVIRGO[:,6]

dataLIGO = np.genfromtxt('data_sensitivity_curves/aligo.txt')
fLIGO=dataLIGO[:,0]
hLIGO=dataLIGO[:,1]
sn_intALIGO = interp1d(fLIGO,hLIGO**2.,bounds_error=False,fill_value=1.0)
minfALIGO = min(fLIGO)
maxfALIGO = max(fLIGO)
def aLIGO():
    return sn_intALIGO, minfALIGO, maxfALIGO

#dataLIGO_design = np.genfromtxt('data_sensitivity_curves/aligo_design.txt')
#fLIGO_design=dataLIGO_design[:,0]
#hLIGO_design=dataLIGO_design[:,1]

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

dataO3 = np.genfromtxt('data_sensitivity_curves/ligo_living_reviews/aligo_O3actual_L1.txt')
#dataO3 = np.genfromtxt('data_sensitivity_curves/o3_h1.txt')
fO3=dataO3[:,0]
hO3=dataO3[:,1]
sn_intO3 = interp1d(fO3,hO3**2.,bounds_error=False,fill_value=1.0)
minfO3 = min(fO3)
maxfO3 = max(fO3)
def O3():
    return sn_intO3, minfO3, maxfO3

dataO4 = np.genfromtxt('data_sensitivity_curves/ligo_living_reviews/aligo_O4high.txt')
fO4=dataO4[:,0]
hO4=dataO4[:,1]
sn_intO4 = interp1d(fO4,hO4**2.,bounds_error=False,fill_value=1.0)
minfO4 = min(fO4)
maxfO4 = max(fO4)
def O4():
    return sn_intO4, minfO4, maxfO4

dataO5 = np.genfromtxt('data_sensitivity_curves/ligo_living_reviews/AplusDesign.txt')
fO5=dataO5[:,0]
hO5=dataO5[:,1]
sn_intO5 = interp1d(fO5,hO5**2.,bounds_error=False,fill_value=1.0)
minfO5 = min(fO5)
maxfO5 = max(fO5)
def O5():
    return sn_intO5, minfO5, maxfO5

dataLIGOeh = np.genfromtxt('data_sensitivity_curves/fig1_aligo_sensitivity.txt')
fLIGOeh=dataLIGOeh[:,0]
hLIGOeh=dataLIGOeh[:,2]
sn_intLIGOeh = interp1d(fLIGOeh,hLIGOeh**2.,bounds_error=False,fill_value=1.0)
minfLIGOeh = min(fLIGOeh)
maxfLIGOeh = max(fLIGOeh)
def LIGOeh():
    return sn_intLIGOeh, minfLIGOeh, maxfLIGOeh

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

dataVO3 = np.genfromtxt('data_sensitivity_curves/ligo_living_reviews/avirgo_O3actual.txt')
#dataO3 = np.genfromtxt('data_sensitivity_curves/o3_h1.txt')
fVO3=dataVO3[:,0]
hVO3=dataVO3[:,1]
sn_intVO3 = interp1d(fVO3,hVO3**2.,bounds_error=False,fill_value=1.0)
minfVO3 = min(fVO3)
maxfVO3 = max(fVO3)
def VO3():
    return sn_intVO3, minfVO3, maxfVO3

dataVO4 = np.genfromtxt('data_sensitivity_curves/ligo_living_reviews/avirgo_O4high_NEW.txt')
fVO4=dataVO4[:,0]
hVO4=dataVO4[:,1]
sn_intVO4 = interp1d(fVO4,hVO4**2.,bounds_error=False,fill_value=1.)
minfVO4 = min(fVO4)
maxfVO4 = max(fVO4)
def VO4():
    return sn_intVO4, minfVO4, maxfVO4

dataVO5 = np.genfromtxt('data_sensitivity_curves/ligo_living_reviews/avirgo_O5high_NEW.txt')
fVO5=dataVO5[:,0]
hVO5=dataVO5[:,1]
sn_intVO5 = interp1d(fVO5,hVO5**2.,bounds_error=False,fill_value=1.)
minfVO5 = min(fVO5)
maxfVO5 = max(fVO5)
def VO5():
    return sn_intVO5, minfVO5, maxfVO5


""" Horizon distance """
datazhorLISA = np.genfromtxt('gw_horizon_data/zhor_max_lisa.txt')
massesLISA=datazhorLISA[:,0]
zhorLISA=datazhorLISA[:,1]
zhormaxLISA = interp1d(massesLISA,zhorLISA,bounds_error=False,fill_value=0.0)

datazhorLISAsky = np.genfromtxt('gw_horizon_data/zhor_max_lisa_sky.txt')
massesLISAsky=datazhorLISAsky[:,0]
zhorLISAsky=datazhorLISAsky[:,1]
zhormaxLISAsky = interp1d(massesLISAsky,zhorLISAsky,bounds_error=False,fill_value=0.0)

datazhorO2 = np.genfromtxt('gw_horizon_data/zhor_max_o2.txt')
massesO2=datazhorO2[:,0]
zhorO2=datazhorO2[:,1]
zhormaxO2 = interp1d(massesO2,zhorO2,bounds_error=False,fill_value=0.0)

datazhorO3 = np.genfromtxt('gw_horizon_data/zhor_max_o3.txt')
massesO3=datazhorO3[:,0]
zhorO3=datazhorO3[:,1]
zhormaxO3 = interp1d(massesO3,zhorO3,bounds_error=False,fill_value=0.0)

datazhorO4 = np.genfromtxt('gw_horizon_data/zhor_max_o4.txt')
massesO4=datazhorO4[:,0]
zhorO4=datazhorO4[:,1]
zhormaxO4 = interp1d(massesO4,zhorO4,bounds_error=False,fill_value=0.0)

datazhorO5 = np.genfromtxt('gw_horizon_data/zhor_max_o5.txt')
massesO5=datazhorO5[:,0]
zhorO5=datazhorO5[:,1]
zhormaxO5 = interp1d(massesO5,zhorO5,bounds_error=False,fill_value=0.0)

datazhorvoyager = np.genfromtxt('gw_horizon_data/zhor_max_voyager.txt')
massesvoyager=datazhorvoyager[:,0]
zhorvoyager=datazhorvoyager[:,1]
zhormaxvoyager = interp1d(massesvoyager,zhorvoyager,bounds_error=False,fill_value=0.0)

datazhorLIGOeh = np.genfromtxt('gw_horizon_data/zhor_max_ligo_eh.txt')
massesLIGOeh=datazhorLIGOeh[:,0]
zhorLIGOeh=datazhorLIGOeh[:,1]
zhormaxLIGOeh = interp1d(massesLIGOeh,zhorLIGOeh,bounds_error=False,fill_value=0.0)

datazhorALIGO = np.genfromtxt('gw_horizon_data/zhor_max_aligo.txt')
massesALIGO=datazhorALIGO[:,0]
zhorALIGO=datazhorALIGO[:,1]
zhormaxALIGO = interp1d(massesALIGO,zhorALIGO,bounds_error=False,fill_value=0.0)

datazhorAplus = np.genfromtxt('gw_horizon_data/zhor_max_aplus.txt')
massesAplus=datazhorAplus[:,0]
zhorAplus=datazhorAplus[:,1]
zhormaxAplus = interp1d(massesAplus,zhorAplus,bounds_error=False,fill_value=0.0)

datazhorET = np.genfromtxt('gw_horizon_data/zhor_max_et.txt')
massesET=datazhorET[:,0]
zhorET=datazhorET[:,1]
zhormaxET = interp1d(massesET,zhorET,bounds_error=False,fill_value=0.0)

datazhorCE = np.genfromtxt('gw_horizon_data/zhor_max_ce.txt')
massesCE=datazhorCE[:,0]
zhorCE=datazhorCE[:,1]
zhormaxCE = interp1d(massesCE,zhorCE,bounds_error=False,fill_value=0.0)

""" Horizon distance (Detector Frame)"""

datazhorLISA_mobs = np.genfromtxt('gw_horizon_data/zhor_max_lisa_mobs.txt')
massesLISA_mobs=datazhorLISA_mobs[:,0]
zhorLISA_mobs=datazhorLISA_mobs[:,1]
zhormaxLISA_mobs = interp1d(massesLISA_mobs,zhorLISA_mobs,bounds_error=False,fill_value=0.0)

datazhorLISAsky_mobs = np.genfromtxt('gw_horizon_data/zhor_max_lisa_sky_mobs.txt')
massesLISAsky_mobs=datazhorLISAsky_mobs[:,0]
zhorLISAsky_mobs=datazhorLISAsky_mobs[:,1]
zhormaxLISAsky_mobs = interp1d(massesLISAsky_mobs,zhorLISAsky_mobs,bounds_error=False,fill_value=0.0)

datazhorO2_mobs = np.genfromtxt('gw_horizon_data/zhor_max_o2_mobs.txt')
massesO2_mobs=datazhorO2_mobs[:,0]
zhorO2_mobs=datazhorO2_mobs[:,1]
zhormaxO2_mobs = interp1d(massesO2_mobs,zhorO2_mobs,bounds_error=False,fill_value=0.0)

#datazhorO3_mobs = np.genfromtxt('gw_horizon_data/zhor_max_o3_mobs.txt')
#massesO3_mobs=datazhorO3_mobs[:,0]
#zhorO3_mobs=datazhorO3_mobs[:,1]
#zhormaxO3_mobs = interp1d(massesO3_mobs,zhorO3_mobs,bounds_error=False,fill_value=0.0)

datazhorO4_mobs = np.genfromtxt('gw_horizon_data/zhor_max_o4_mobs.txt')
massesO4_mobs=datazhorO4_mobs[:,0]
zhorO4_mobs=datazhorO4_mobs[:,1]
zhormaxO4_mobs = interp1d(massesO4_mobs,zhorO4_mobs,bounds_error=False,fill_value=0.0)

datazhorO5_mobs = np.genfromtxt('gw_horizon_data/zhor_max_o5_mobs.txt')
massesO5_mobs=datazhorO5_mobs[:,0]
zhorO5_mobs=datazhorO5_mobs[:,1]
zhormaxO5_mobs = interp1d(massesO5_mobs,zhorO5_mobs,bounds_error=False,fill_value=0.0)

#datazhorvoyager_mobs = np.genfromtxt('gw_horizon_data/zhor_max_voyager_mobs.txt')
#massesvoyager_mobs=datazhorvoyager_mobs[:,0]
#zhorvoyager_mobs=datazhorvoyager_mobs[:,1]
#zhormaxvoyager_mobs = interp1d(massesvoyager_mobs,zhorvoyager_mobs,bounds_error=False,fill_value=0.0)

#datazhorLIGOeh_mobs = np.genfromtxt('gw_horizon_data/zhor_max_ligo_eh_mobs.txt')
#massesLIGOeh_mobs=datazhorLIGOeh_mobs[:,0]
#zhorLIGOeh_mobs=datazhorLIGOeh_mobs[:,1]
#zhormaxLIGOeh_mobs = interp1d(massesLIGOeh_mobs,zhorLIGOeh_mobs,bounds_error=False,fill_value=0.0)

#datazhorALIGO_mobs = np.genfromtxt('gw_horizon_data/zhor_max_aligo_mobs.txt')
#massesALIGO_mobs=datazhorALIGO_mobs[:,0]
#zhorALIGO_mobs=datazhorALIGO_mobs[:,1]
#zhormaxALIGO_mobs = interp1d(massesALIGO_mobs,zhorALIGO_mobs,bounds_error=False,fill_value=0.0)

#datazhorAplus_mobs = np.genfromtxt('gw_horizon_data/zhor_max_aplus_mobs.txt')
#massesAplus_mobs=datazhorAplus_mobs[:,0]
#zhorAplus_mobs=datazhorAplus_mobs[:,1]
#zhormaxAplus_mobs = interp1d(massesAplus_mobs,zhorAplus_mobs,bounds_error=False,fill_value=0.0)

datazhorET_mobs = np.genfromtxt('gw_horizon_data/zhor_max_et_mobs.txt')
massesET_mobs=datazhorET_mobs[:,0]
zhorET_mobs=datazhorET_mobs[:,1]
zhormaxET_mobs = interp1d(massesET_mobs,zhorET_mobs,bounds_error=False,fill_value=0.0)

datazhorCE_mobs = np.genfromtxt('gw_horizon_data/zhor_max_ce_mobs.txt')
massesCE_mobs=datazhorCE_mobs[:,0]
zhorCE_mobs=datazhorCE_mobs[:,1]
zhormaxCE_mobs = interp1d(massesCE_mobs,zhorCE_mobs,bounds_error=False,fill_value=0.0)


