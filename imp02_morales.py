# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

# %%
# Read data from text file into DataFrame
leafRiverDaily = pd.read_fwf('LeafRiverDaily.txt',
                            names=['precip',        # areal avg precip
                                   'pET',           # areal avg potential evap
                                   'Q'])            # streamflow @ catchment outflow

# Splice dataframe to include 1st water year only
lr_yearone = leafRiverDaily.iloc[0:365]                # "wy" - water year; "dd" - daily 

# Select the start and end of period of observation
periodStart = 0; periodEnd = 365
time = np.arange(periodStart, periodEnd)

# Select the observations according to the chosen period of observation
PPobs = leafRiverDaily.precip[time]; PEobs = leafRiverDaily.pET[time]; QQobs = leafRiverDaily.Q[time]

# Add datetimeindex to dataframe, offsetting default year by 22 years
#yearone_date = pd.to_datetime(lr_yearone.index, 
#                             unit='D').map(lambda x: x.replace(year = x.year - 22))
#lr_yearone = lr_yearone.set_index(yearone_date)

# %%
# PART ONE: plotting the data onto four subplots - precip, et, Q, & Q

x = list(lr_yearone.index.values)
y = lr_yearone['precip']

fig, axs = plt.subplots(4,1, figsize=(12,8))
fig.set_facecolor('whitesmoke')

axs[0].bar(x, y, width=1.25, label='P as bar plot')
axs[0].set_ylim(100, 0)
axs[0].set_xlim(0, 365)
axs[0].set_ylabel('PP (mm)', fontweight='bold')
axs[0].grid(True)
axs[0].set_title('Leaf River Daily Climatic Data (WY 1948)', fontweight='bold')
axs[0].legend(loc='lower right')
axs[0].tick_params(axis='both', direction='in')

y = lr_yearone['pET']

axs[1].plot(x, y, label='ET as line plot')
axs[1].plot(x,y, 'ok', mec='k', mfc='k', markersize=1, label='ET as dot')
axs[1].set_ylim(0,8)
axs[1].set_xlim(0, 365)
axs[1].set_ylabel('ET (mm)', fontweight='bold')
axs[1].grid(True)
axs[1].legend(loc='upper right')
axs[1].tick_params(axis='both', direction='in')

y = lr_yearone['Q']

axs[2].plot(x, y, 'ok', markersize='2.5', label='Q as o-symbol')
axs[2].set_ylim(0,30)
axs[2].set_xlim(0, 365)
axs[2].set_ylabel('Q (cms)', fontweight='bold')
axs[2].fill_between(x, y, 0, alpha=0.4, label='Q as area plot', edgecolor='k')
axs[2].legend(loc='upper right')
axs[2].tick_params(axis='both', direction='in')

#axs[3].plot(x, y, '-ok', mec='r', mfc='r', markersize='2.5')
axs[3].semilogy(x,y, 'k', label='Q as line plot')
axs[3].semilogy(x,y,'ok', mec='r', mfc='r', markersize='2.5', label='Q as dot markers')
axs[3].set_xlim(0, 365)
axs[3].set_ylabel('Q (cms)', fontweight='bold')
axs[3].set_xlabel('Time (days)]', fontweight='bold')
axs[3].legend(loc='upper left')
axs[3].tick_params(axis='both', direction='in')

plt.savefig('imp02-p1_morales')
plt.show()

# %%
# Define the linear reservoir model:
def linRes(XXin, PP, PE, Kpar):
    '''This function simulates a one-parameter linear reservoir.

    Parameters
    ----------
    XXin : int
       soil moisture, input state variable
    PP : float
       precipitation, input variable
    PE : float
       potential evapotranspiration, input variable
    Kpar : float
       the available streamflow constant

    Returns
    ----------
    QQ : float
       generated runoff
    ET : float
       amt of evaporated water, limited by available energy or water
    XXout : float
       new state variable 
    '''
    QQ = Kpar*XXin                 # set streamflow as a function of XX and Kpar
    ET = min(PP, PE)               # set ET as limited by available water (XX) or energy (PE)
    XXout = XXin - QQ - ET + PP    # set XX after timestep as fxn of XXin, runoff, et, and precip
    return(QQ, ET, XXout)          # return the output variables; runoff, et, and new state variable

# %%
fig, ax = plt.subplots(figsize=(20,5))
fig.set_facecolor('whitesmoke')

Kpars = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.25]
nKpars = len(Kpars)
KparsIndex = np.arange(0, nKpars, 1)

ax.plot(time, QQobs, "or", label='Observed QQ', markersize=3)
ax.grid('k-.-')
ax.set_ylabel('QQ (cms)', fontweight='bold')
ax.set_xlabel('Time (days)', fontweight='bold')
ax.set_title('Predicted streamflow (QQ) vs. observations\nWater Year 1948', fontweight='bold')
ax.set_xlim(periodStart,periodEnd)
ax.set_ylim(0, 1.5*QQobs.max())

colors = ['orangered','gold','chartreuse','turquoise','royalblue','fuchsia']

for i in KparsIndex:
     Kpar = Kpars[i]
     XX = np.zeros(365+1); QQsim = np.zeros(365); ETsim = np.zeros(365)
     lbl = 'Predicted QQ; Kpar = {}'.format(str(Kpar))
     color = colors[i]

     for t in time:
         [QQsim[t], ETsim[t], XX[t+1]] = linRes(XX[t], PPobs[t], PEobs[t], Kpar)     
              
     ax.plot(time, QQsim, '-', label=lbl, color=color)
plt.legend()

# %
# %%
