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
plt.savefig('imp02-p2_linRes_morales')

# %%
# Create a function to describe the Nash-Cascade model

def nashCasc(PP, NRes, XXin, Kpar):
    Res = np.arange(0, NRes, 1)
    UU = PP
    PE = 0.0
    XXout = np.zeros(NRes)
    QQind = np.zeros(NRes)

    for i in Res:
      [QQind[i], ET, XXout[i]] = linRes(XXin[i], UU, PE, Kpar[i])
      UU = QQind[i]

    QQout = UU
    return [QQout, XXout]

# %%

Kpar = 0.01; Nres = 3

PX = np.zeros(365)
XXin = np.zeros(Nres)
QQsim = np.zeros(365)
for t in time:
    PX[t] = PPobs[t] - np.minimum(PPobs[t], PEobs[t])
    [QQsim[t], XXout] = nashCasc(PX[t], Nres, XXin, Kpar)
    XXin = XXout

fig, ax = plt.subplots()

ax.plot(time,QQsim)
# %%
# Represent system as a series of reservoirs using a Nash-Cascade model
fig, ax = plt.subplots(figsize=(20,5))
fig.set_facecolor('whitesmoke')

ax.plot(time, QQobs, 'or', markersize=3, label='Observed QQ')
ax.set_xlim(periodStart, periodEnd)
ax.grid(True)

nbins = 3; bin = 0; Kpar = 0.1
XX = np.zeros(365+1); QQsim = np.zeros(365); ETsim = np.zeros(365)

while bin <= nbins:
    for t in time:
       [QQsim[t], ETsim[t], XX[t+1]] = linRes(XX[t], PPobs[t], PEobs[t], Kpar)

    XX[0] = XX[-1]
    print(XX[-1])
    bin += 1
ax.plot(time, QQsim, '-', label='trial')
plt.legend()

# %%
# def LinRes(XXin, PP, PE, Kpar):                 # Define LinRes function subprogram       
#     QQ = Kpar*XXin;                             # Compute QQ
#     ET = min(PP,PE);                            # Compute ET
#     XXout = XXin - QQ - ET + PP;                 # Update state variable
#     return [QQ, ET, XXout]                       # Return compute quantities

def NashCasc(PP, NRes, XXin, Kpar):           # Define NashCasc function subprogram
    Res   = np.arange(0,NRes,1);              # Set Loop vector on tanks (integer values) 
    UU    = PP;                               # UU is set to be input to first tank
    PE    = 0.0;                              # PE is set to be zero
    XXout = np.zeros(NRes);                   # Initialize array to recieve XXout
    QQind = np.zeros(NRes);                   # Initialize array to recieve QQint (individual tank outflows)

    for i in Res: # Begin loop on tanks in series
        [QQind[i], ET, XXout[i]] = linRes(XXin[i],UU,PE,Kpar[i]); # Call LinRes subprogram
        UU = QQind[i];                        # Set UU (inflow to next tank) to be 
                                              # outflow from previous tank
    # End loop on tanks in series    
    QQout = UU;                               # Get output from last tank
    return [QQout, XXout];                    # Return computed quantities

# %%


colors = ['orangered','gold','chartreuse','turquoise','royalblue','fuchsia']

Kpars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
nKpars = len(Kpars)
KparsIndex = np.arange(0, nKpars, 1)

nashRes = [1, 2, 3, 4, 5]
nashResLength = len(nashRes)
nResIndex = np.arange(0, nashResLength, 1)

fig, ax = plt.subplots(nashResLength, 1, figsize=(15,20), sharex=True)
fig.set_facecolor('whitesmoke')
# fig.suptitle('Nash-Cascade predicted streamflow (QQ) vs. observations\nWater Year 1948', fontweight='bold')

for k in nResIndex:

   for i in KparsIndex:
      
      PX = np.zeros(365);                                  # Initialize PX
      NRes = nashRes[k]                           # Get NRes value (as integer)
      Kpar = np.ones(NRes)*Kpars[i]                     # Set all tanks to same Kpar value
      XXin = np.zeros(NRes);                               # Initialize XXin
      QQsim = np.zeros(365);                               # Initialize QQsim array
      for t in time: # Run loop on Time
         PX[t] = PPobs[t] - np.minimum(PPobs[t],PEobs[t]);      # Compute PX for time step
         [QQsim[t], XXout] = NashCasc(PX[t], NRes, XXin, Kpar); # Run Nash Cascade for one time step
         XXin = XXout; 


      ax[k].plot(time, QQsim, color=colors[i], label='Predicted QQ; Kpar={}'.format(str(Kpar[0])))
      ax[k].text(0.06, 0.9, 'n reservoir={}'.format(str(k+1)), horizontalalignment='center', verticalalignment='center', transform=ax[k].transAxes, bbox=dict(facecolor='white', edgecolor='k', pad=3.0))

   ax[k].plot(time, QQobs, 'or', markersize=2.5, label='Observed QQ')
   ax[k].set_ylabel('QQ (cms)', fontweight='bold')
   # ax[k].set_title('n={}'.format(NRes), fontweight='bold')
   ax[k].grid(True)
   ax[k].set_xlim(periodStart, periodEnd)
   
   plt.legend()
ax[k].set_xlabel('Time (days)', fontweight='bold')
ax[0].set_title('Nash-Cascade predicted streamflow (QQ) vs. observations\nWater Year 1948', fontweight='bold')
plt.tight_layout
# %%
