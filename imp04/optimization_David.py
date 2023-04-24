import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import catchment_model
import performance_metrics
import objective_function as OF


# Load Leaf River Catchment Data as a Data Frame ========================================
LRData = pd.read_csv('/Users/omidzandi/Desktop/PhD/Courses/Systems_approach_to_hydrological_modeling/individual_project/report_4/codes/LeafRiverDaily.csv');

# Get number of data points
Ndata = LRData.shape[0];
Nvars = LRData.shape[1];

# Choose Simulation Time Period & Set Up Time Vector ====================================
Period_Start = 0;  Period_End = 365*3;          # Select Time Period to work with
Time = np.arange(Period_Start,Period_End,1);    # Set up Time vector (of integer values) 
NTime = len(Time);                              # Get length of Time vector

# Set num. of initial time steps to use as Spin-Up ======================================
# (Metrics are not computed on this period)
SpinUp = 90;                                    # Select 90 day spin-up period

# Get Data for the Desired Period =======================================================
PPobs = LRData.PP[Time].to_numpy();             # CONVERT FROM PANDA DATA FRAMES TO
PEobs = LRData.PE[Time].to_numpy();             # NUMPY ARRAYS TO AVOID INDEXING ISSUES
QQobs = LRData.QQ[Time].to_numpy();

# Determine Maximum Values for Each Variable ============================================
PPobs_max = np.max(PPobs);                      # Get max values for setting plot ranges
PEobs_max = np.max(PEobs); 
QQobs_max = np.max(QQobs);

# Specify Model Parameters ==============================================================
NRes   = 2; ResIdx = np.arange(0,NRes,1);       # Specify Number of Nash Cascade Tanks

# Specify Model Parameter Values for Simulation (initial guess of the previous report)
#Theta_C1  = 70.0;    # Upper Soil Zone Capacity
#Theta_P1  = 1.0;     # Evapotranspiration Parameter (Reducing value increases ET)  
#Theta_K12 = 0.20;    # Drainage to Lower Soil Zone (Increasing value increases Drainage) 
#Theta_K13 = 0.10;    # Interflow Parameter (Increasing value increases Interflow)  
#Theta_K2Q = 0.0001;  # Baseflow Parameter (Increasing value makes baseflow recessions steeper) 
#Theta_K2Q = -4;  # Baseflow Parameter (Increasing value makes baseflow recessions steeper) 
#Theta_K3Q = 0.6;     # Channel Routing Parameter (Increasing value reduces temporal dispersion)



# Specify Model Parameter Values for Simulation (manual calibration)
Theta_C1  = 55;    # Upper Soil Zone Capacity
Theta_P1  = 0;     # Evapotranspiration Parameter (Reducing value increases ET)  
Theta_K12 = 0.3;    # Drainage to Lower Soil Zone (Increasing value increases Drainage) 
Theta_K13 = 0.10;    # Interflow Parameter (Increasing value increases Interflow)  
#Theta_K2Q = 0.0001;  # Baseflow Parameter (Increasing value makes baseflow recessions steeper) 
Theta_K2Q = -4;  # Baseflow Parameter (Increasing value makes baseflow recessions steeper) 
Theta_K3Q = 0.6;     # Channel Routing Parameter (Increasing value reduces temporal dispersion)


# Specify Model Parameter Values for Simulation (sth between Mohammad and )
#Theta_C1  = 55;    # Upper Soil Zone Capacity
#Theta_P1  = 0.5;     # Evapotranspiration Parameter (Reducing value increases ET)  
#Theta_K12 = 0.3;    # Drainage to Lower Soil Zone (Increasing value increases Drainage) 
#Theta_K13 = 0.10;    # Interflow Parameter (Increasing value increases Interflow)  
#Theta_K2Q = 0.0001;  # Baseflow Parameter (Increasing value makes baseflow recessions steeper) 
#Theta_K2Q = -3;  # Baseflow Parameter (Increasing value makes baseflow recessions steeper) 
#Theta_K3Q = 0.6;     # Channel Routing Parameter (Increasing value reduces temporal dispersion)


# Specify Model Parameter Values for Simulation (Mohammad Initial) 
#Theta_C1  = 60;    # Upper Soil Zone Capacity
#Theta_P1  = 0.2;     # Evapotranspiration Parameter (Reducing value increases ET)  
#Theta_K12 = 0.01;    # Drainage to Lower Soil Zone (Increasing value increases Drainage) 
#Theta_K13 = 0.1;    # Interflow Parameter (Increasing value increases Interflow)  
##Theta_K2Q = 0.0001;  # Baseflow Parameter (Increasing value makes baseflow recessions steeper) 
#Theta_K2Q = -4;  # Baseflow Parameter (Increasing value makes baseflow recessions steeper) 
#Theta_K3Q = 0.6;     # Channel Routing Parameter (Increasing value reduces temporal dispersion)

Pars = np.zeros(6)

Pars[0] = Theta_C1;  Pars[1] = Theta_P1;  Pars[2] = Theta_K12;
Pars[3] = Theta_K13; Pars[4] = Theta_K2Q; Pars[5] = Theta_K3Q;

# The plausible range of each parameter
#Theta_C1_Range = [10,200]
#Theta_P1_Range = [0,1]
#Theta_P1_Range = [0,2]
#Theta_K12_Range = [0,0.5]
#Theta_K13_Range = [0,1]
#Theta_K2Q_Range = [-6,-1]
#Theta_K3Q_Range = [0,1]

n_iters = 1000

# perform the search
result = minimize(OF.obj_func, Pars, args = (PPobs, PEobs, QQobs, NRes), 
                  bounds= ((10,200), (0,2), (0,0.5), (0,1), (-4,-1), (0,1)), 
                  method='nelder-mead', options={'disp': True, 'return_all': True, 'maxiter':n_iters})

# Mohammad
#result = minimize(OF.obj_func, Pars, args = (PPobs, PEobs, QQobs, NRes), 
#                  bounds= ((10,200), (0.1,1.1), (0,0.5), (0.1,1), (-6,-1), (0.1,0.9)), 
#                 method='nelder-mead', options={'disp': True, 'return_all': True, 'maxiter':n_iters})

#result = minimize(OF.obj_func, Pars, args = (PPobs, PEobs, QQobs, NRes), 
#                  method='nelder-mead', options={'disp': True, 'return_all': True, 'maxiter':n_iters})

# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = OF.obj_func(solution, PPobs, PEobs, QQobs, NRes)
print('Solution: f(%s) = %.5f' % (solution, 1/evaluation))

len_output_iters = len(result['allvecs'])
iter_KGEss = np.zeros(len_output_iters)
for index, params in enumerate(result['allvecs']):
    # Becareful not to forget the inversion of KGEss
    iter_KGEss[index] = 1/(OF.obj_func(params, PPobs, PEobs, QQobs, NRes))

############################## Iteration KGE Plot 
fig_width = 14; fig_height = 4
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width,fig_height), dpi=250)

#ax.set_xlim(Period_Start, Period_End);                   # Fix xaxis limits
ax.set_xlabel('Iteration');                           # Set the x-axis label
ax.set_ylabel('KGEss');                               # Set the x-axis label
#ax.set_title('');           # Add plot title
ax.grid('k--');                                         # Turn on grid lines
B = [0.98*np.min(iter_KGEss), 1.02*np.max(iter_KGEss)];                               # Get the y-axis limits 
ax.set_ylim(B);                                         # Set the y-axis limits                             # Choose line color
ax.plot(np.arange(1, len_output_iters+1), iter_KGEss, 'b-');         # Plots the Observed Flows
plt.savefig('/Users/omidzandi/Desktop/PhD/Courses/Systems_approach_to_hydrological_modeling/individual_project/report_4/codes/_KGE_iteration_from_manual.jpg', bbox_inches="tight")


# Converting the list of optimum parameters to numpy array
iter_opt_params = np.zeros((len_output_iters, 6))
for index, params in enumerate(result['allvecs']):
    iter_opt_params[index,:] = params

# Transforming the fifth column to float
iter_opt_params[:,-2] = 10 ** iter_opt_params[:,-2]

################################# Iteration Parameter Plot
fig_width = 14; fig_height = 4
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(fig_width,fig_height), sharex=True, dpi=250)
ax = np.ravel(ax)
#fig.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

param_names = ['Theta_C1', 'Theta_P1', 'Theta_K12', 'Theta_K13', 'Theta_K2Q', 'Theta_KQ']

for i in np.arange(0, iter_opt_params.shape[1]):


    #ax.set_title('');           # Add plot title
    ax[i].grid('k--');                                         # Turn on grid lines

    ax[i].plot(np.arange(1, len_output_iters+1), iter_opt_params[:,i], 'b-')

    B = [0.98*np.min(iter_opt_params[:,i]), 1.02*np.max(iter_opt_params[:,i])]; 
    ax[i].set_ylim(B);                                         # Set the y-axis limits                             # Choose line color
    #ax[i].legend(loc=2);

    #ax[i].set_xlabel('Iteration');                           # Set the x-axis label
    ax[i].set_ylabel(param_names[i]);
    
    if i == 4:
        ax[i].ticklabel_format(style='sci', axis='y', useOffset=False, scilimits=(0,0))


fig.text(0.5, 0.01, 'Iteration', ha='center')
fig.tight_layout()

plt.savefig('/Users/omidzandi/Desktop/PhD/Courses/Systems_approach_to_hydrological_modeling/individual_project/report_4/codes/_one_subplots_for_each_parameter_from_manual.jpg', bbox_inches="tight")

a = 1
print('Finish')