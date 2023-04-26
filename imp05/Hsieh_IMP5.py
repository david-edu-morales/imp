# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
# loading LRC data

filename = 'LeafRiverDaily.txt'
path = "/Users/dnahh/Documents/HWRS_528(Systems)/Project/"
filepath = os.path.join(path, filename)
print(os.getcwd())
print(filepath)

# read the data into pandas dataframe
LRCdata = pd.read_table(filepath, sep=' ', skipinitialspace = True)

# Making columns for my data and adding back in my first row of data which mysteriously
# wont load. I also resorted the index.
LRCdata.columns = ['rainfall [mm/day]', 'Potential ET [mm/day]', 'Streamflow [mm/day]']
LRCdata.loc[-1] = [0 , 4.6004, 0.289951]
LRCdata.index = LRCdata.index +1 
LRCdata = LRCdata.sort_index()

# %%
# Defining all of my functions

def USS(OK12, XX1, OK13, PP, OC1, PE, OP1):
    """Fuction to compute upper soil system overland flow and recharge"""

    IF = OK13 * XX1
    PX = max(0, PP + XX1 - OC1)
    OF = PX + IF
    XX1a = XX1 - IF
    AE = min((PE * ((XX1a/OC1)**OP1)), XX1a)
    XX1b = XX1a - AE
    RG = OK12 * XX1b
    XX1c = XX1b - RG
    XX1_1 = XX1c + (PP - PX)
    return RG, OF, XX1_1

# Lower Soil System
def LSS(RG, OK2Q, XX2):
    """Function to determine baseflow in lower soil system"""
    
    BF = OK2Q * XX2
    XX2_1 = XX2 - BF + RG
    return BF, XX2_1

# Channel System/Nash Cascade
def Nash(OK3Q, XX31, XX32, OF):
    """Fucnction to determine surface flow"""

    Y = OK3Q * XX31
    SF = OK3Q * XX32
    XX31_1 = XX31 - Y + OF
    XX32_1 = XX32 - SF + Y
    return SF, XX31_1, XX32_1

def KGEssFunc(P, Metricflag): # Our new 'QuadFunc'. It calculates our KGEss given our 6 parameters. 
    """Function to determine KGEss. This comes from my IMP3. P is an array of numbers.
    The only thing this function returns is KGEss, our new 'Quad func' 
    
    P is our matrix of parameters and MetricFlag is our 'flag' to tell the function what to print"""

    OC1 = P[0] #P[0] is the first number in the array "P"
    OK12 = P[1] #P[1] is the second number in the array "P"
    OK13 = P[2]
    OP1 =  P[3]
    OK2Q = P[4] #P[4] is the fifth number in the array "P"
    OK3Q = P[5]
    XX1 = 0 # Water in Upper Soil
    XX2 = float(LRCdata.iloc[0, 2])/OK2Q # water in subsurface
    XX31 = 0.07 # Water in first Nash Cascade
    XX32 = 0.5 # Water in second Nash Cascade 

    SFarray= np.array([]) # Making empty arrays we populate below when I run my USS, LSS, and Nash functions
    OFarray= np.array([])
    BFarray= np.array([])
    QQarray= np.array([])
    XX1array= np.array([])
    XX2array= np.array([])
    XX32array= np.array([])

    #This calculates recharge, overlandflow, state variables (XX1, XX2, XX31, XX32), baseflow, surface flow, and our streamflow
    for k in range(0,1095):
        RG, OF, XX1 = USS(OK12, XX1, OK13, float(LRCdata.iloc[k, 0]), OC1, float(LRCdata.iloc[k, 1]), OP1)
        BF, XX2 = LSS(RG, OK2Q, XX2)
        SF, XX31, XX32 = Nash(OK3Q, XX31, XX32, OF)
        QQ = SF + BF
        SFarray= np.append(SFarray, SF)
        OFarray= np.append(OFarray, OF)
        BFarray= np.append(BFarray, BF)
        QQarray= np.append(QQarray, QQ)
        XX1array= np.append(XX1array, XX1)
        XX2array= np.append(XX2array, XX2)
        XX32array= np.append(XX32array, XX32)
    
    #Calculating MSE
    tempsum = 0
    for n in range(92,1095):
        tempsum = tempsum + (QQarray[n] - LRCdata.iloc[n,2])**2 
    MSE = (1/1002) * tempsum

    #Calculating mean of QQ_obs
    tempsum2 = 0
    for n in range(92,1095):
        tempsum2 = tempsum2 + LRCdata.iloc[n,2]
    uQQ_obs = (1/1002) * tempsum2

    #Calculating NSE
    tempsum3 = 0
    for n in range(92,1095):
        tempsum3 = tempsum3 + (LRCdata.iloc[n,2] - uQQ_obs)**2
    o2QQ_obs = (1/(1002-1)) * tempsum3

    NSE = 1 - (MSE/o2QQ_obs)
    
    #Calculating mean QQ_sim
    tempsum4 = 0
    for n in range(92,1095):
        tempsum4 = tempsum4 + QQarray[n]
    uQQ_sim = (1/1002) * tempsum4

    #Calculating standard deviation of QQ_sim
    tempsum5 = 0
    for n in range(92,1095):
        tempsum5 = tempsum5 + (QQarray[n] - uQQ_sim)**2
    o2QQ_sim = (1/(1002-1)) * tempsum5

    #Calculating alpha and beta
    alpha = (o2QQ_sim)**(1/2)/(o2QQ_obs)**(1/2)
    beta = uQQ_sim/uQQ_obs
    
    #Calculating rho
    tempsum6 = 0
    for n in range(92,1095):
        tempsum6 = tempsum6 + ((QQarray[n]-uQQ_sim)/(o2QQ_sim**(1/2)))* \
            ((LRCdata.iloc[n,2]-uQQ_obs)/o2QQ_obs**(1/2))
    rho = (1/(1002-1)) * tempsum6
    
    #Calculating KGE
    KGE = 1 - ((1-alpha)**2 + (1-beta)**2 + (1-rho)**2)**(1/2)
    
    #Calculating KGEss
    KGEss = (KGE + 0.4142)/(2**(1/2))
    
    if Metricflag == 0: #This tells us what we want to return. I don't really need metricflag but am reusing from IMP4
        return KGEss
    else:
        return QQarray, alpha, beta, rho, NSE, KGEss 

# %%
# Uncertainty Analysis
  
ParRng = [[10, 300], [0, 0.5], [0.2, 0.9], [0.5, 1.5], [0.0001, 0.1], [0.1, 0.9]] # These are the ranges for our 6 parameters
ParRng = np.array(ParRng).astype(float) # Turn ParRng into an array

NPar = 6
ParIni = np.zeros([NPar,1]).astype(float)

QQarray_list = []
alpha_list = []
beta_list = []
rho_list = []
NSE_list = []
KGEss_list = []
uncertainty_list = []


for i in range(200):
    Rand = np.random.uniform(size = 6) # randomly select inital starting point
    ParIni[0] = ParRng[0,0] + Rand[0] * (ParRng[0,1] - ParRng[0,0]) # Generating random point for P0
    ParIni[1] = ParRng[1,0] + Rand[1] * (ParRng[1,1] - ParRng[1,0]) # Generating random point for P1
    ParIni[2] = ParRng[2,0] + Rand[2] * (ParRng[2,1] - ParRng[2,0]) # Generating random point for P2
    ParIni[3] = ParRng[3,0] + Rand[3] * (ParRng[3,1] - ParRng[3,0]) # Generating random point for P3
    ParIni[4] = ParRng[4,0] + Rand[4] * (ParRng[4,1] - ParRng[4,0]) # Generating random point for P4
    ParIni[5] = ParRng[5,0] + Rand[5] * (ParRng[5,1] - ParRng[5,0]) # Generating random point for P5
    QQarray, alpha, beta, rho, NSE, KGEss  = KGEssFunc(ParIni, 1)
    residualQQ = QQarray - LRCdata.iloc[0:1095,2].tolist()
    uncertainty = residualQQ/LRCdata.iloc[0:1095,2].tolist()
    QQarray_list.append(QQarray)
    alpha_list = np.append(alpha_list, alpha)
    beta_list = np.append(beta_list, beta)
    rho_list = np.append(rho_list, rho)
    NSE_list = np.append(NSE_list, NSE)
    KGEss_list = np.append(KGEss_list, KGEss)
    uncertainty_list.append(uncertainty)


# %%
# Graph of my model simulation showing ensembles of paramteter values 

QQ_95 = np.quantile(QQarray_list, 0.95, axis=0, keepdims=True)
QQ_05 = np.quantile(QQarray_list, 0.05, axis=0, keepdims=True)
tQQ_95 = np.transpose(QQ_95)
tQQ_05 = np.transpose(QQ_05)

fig, ax1 = plt.subplots(squeeze = True, facecolor = 'lightgray', figsize = (15,5))
ax1.bar(list(range(365, 730)),LRCdata.iloc[365:730,0].tolist(), label = 'precip', color='b')
ax1.set_ylim(0,150)
ax1.set(ylabel= "PP (mm)")
ax1.invert_yaxis()
ax2 = ax1.twinx()
#ax2.scatter(list(range(0,1095)),tQQ_95, c = 'k', linewidth = 0.01, alpha = 0.5)
ax2.plot(list(range(0,1095)),tQQ_95,  '-k', linewidth = 0.8)
#ax2.scatter(list(range(0,1095)),tQQ_05, c = 'r', linewidth = 0.01, alpha = 0.5)
ax2.plot(list(range(0,1095)),tQQ_05,  '-k', linewidth = 0.8)
ax2.set(title = 'Quantile of 100 random streamflows', ylabel = 'streamflow', xlabel = 'Time (days)')
ax2.grid(linestyle = ':', linewidth = 1, axis = 'both') 
ax2.plot(LRCdata.iloc[0:1095]['Streamflow [mm/day]'], 'or', label = "Observed Streamflow", markerfacecolor = 'none', markersize = 3)
ax2.fill_between(list(range(0,1095)), tQQ_95[:,0], tQQ_05[:,0], facecolor = 'skyblue', label = "Dispersion between quantile 95% and 5%")
ax2.legend(loc='upper right')
ax2.set_xlim(365,730)


# %%
# I think this makes my density plot?? No idea. 

pd.DataFrame(alpha_list).plot(kind='density', title = "alpha")
pd.DataFrame(beta_list).plot(kind='density', title = "beta" )
pd.DataFrame(rho_list).plot(kind='density', title = "rho")
pd.DataFrame(NSE_list).plot(kind='density', title = "NSE")
pd.DataFrame(KGEss_list).plot(kind='density', title = "KGEss")

# %%
# Ensemble Relative Time-Series Plot

uncertainty_95 = np.quantile(uncertainty_list, 0.95, axis=0, keepdims=True)
uncertainty_05 = np.quantile(uncertainty_list, 0.05, axis=0, keepdims=True)
tuncertainty_95 = np.transpose(uncertainty_95)
tuncertainty_05 = np.transpose(uncertainty_05)

fig, ax3 = plt.subplots(squeeze = True, facecolor = 'lightgray', figsize = (15,5))
ax3.plot(list(range(0,1095)), tuncertainty_95,  '-k', linewidth = 0.8)
ax3.plot(list(range(0,1095)), tuncertainty_05,  '-k', linewidth = 0.8)
ax3.set (title = 'Uncertainty', ylabel = 'Modeled-Obs/Obs', xlabel = 'Time (days)')
ax3.grid(linestyle = ':', linewidth = 1, axis = 'both') 
ax3.fill_between(list(range(0,1095)), tuncertainty_95[:,0], tuncertainty_05[:,0], facecolor = 'skyblue')
ax3.legend(loc='upper right')
ax3.set_xlim(365,730)


# %%
