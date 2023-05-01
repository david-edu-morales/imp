# %%
#==== Import python packages =====================================================
import numpy as np
import pandas as pd

# %%
#==== Linear reservoir module (for one time step) ================================

def LinRes(Xin, U, Kpar):                   # Define LinRes function subprogram

    Y = Kpar*Xin                            # Compute QQ
    Xout = Xin - Y + U                      # Update state variable

    return [Y, Xout]                        # Return computed quantities

#==== Channel routing module (for one time step) ====================================================

def NashCasc(OF, NRes, XX3in, Theta_K3Q):   # Define NashCasc function subprogram

    ResIdx = np.arange(0, NRes, 1)          # Set loop vector on tanks (int values)
    UU     = OF                             # UU is set to be input to first tank
    XX3out = np.zeros(NRes)                 # Initialize array to recieve XX3out
    YY     = np.zeros(NRes)                 # Initialize array to recieve YY (individual tank outflows)

    for i in ResIdx:
        [YY[i], XX3out[i]] = LinRes(XX3in[i], UU, Theta_K3Q)
        UU = YY[i]                          # set UU (inflow to next tank) to be outflow from previous tank
    SF = UU                                 # Get output from last tank

    return [SF, XX3out]                     # Return computed quantities (surface flow and state variable of last tank)

#==== Lower Soil Zone Module (for one time step) ======================================================

def LowerSoilZone(XX2in, RG, Theta_K2Q):    # Calculate lower soil zone subprocesses using LinRes

    [BF, XX2out] = LinRes(XX2in, RG, Theta_K2Q) # Use Recharge as input for LSZ from upper soil zone tank
    return [BF, XX2out]

#==== Upper Soil Zone Module (for one time step) ======================================================

def UpperSoilZone(XX1in, PP, PE, Theta_C1, Theta_P1, Theta_K13, Theta_K12):
    
    PX   = np.maximum(0, PP + XX1in - Theta_C1)   # Determines excess precipitation, part 1 of OF
    IF   = Theta_K13*XX1in                        # Determines lateral drainage, part 2 of OF
    OF   = PX + IF                                # Overland flow that will become the input for NashCasc
    XX1a = XX1in - IF                             # Remainder of water content (Method of Operator Splitting)
    AE   = np.minimum(XX1a, PE*(XX1a/Theta_C1)**Theta_P1) # Actual evaporation limited by PE or water content
    XX1b = XX1a - AE                              # Remainder of water content 
    RG   = Theta_K12*XX1b                         # Recharge
    XX1c = XX1b - RG                              # Remainder of water content
    XX1out = XX1c + PP - PX                       # State variable at end of time step

    return [OF, PX, IF, AE, RG, XX1out]

#==== Catchment Model (for one time step) ======================================================

def CatchModel(Inputs, InStates, Pars, NRes):

    # Get input values for time step
    PP = Inputs[0]
    PE = Inputs[1]

    # Get state values at beginning of time step
    XX1in  = InStates[0]
    XX2in  = InStates[1]
    XX3in  = np.zeros(NRes)
    ResIdx = np.arange(0, NRes, 1)
    for i in ResIdx:
        XX3in[i] = InStates[i+2]

    # Get Par values
    Theta_C1  = Pars[0]
    Theta_P1  = Pars[1]
    Theta_K12 = Pars[2]
    Theta_K13 = Pars[3]
    Theta_K2Q = Pars[4]
    Theta_K3Q = Pars[5]

    # Initialize Output array
    Outputs   = np.zeros(2)
    OutStates = np.zeros(NRes+2)
    IntFluxes = np.zeros(6)

    # Run model components
    [OF, PX, IF, AE, RG, XX1out] = UpperSoilZone(XX1in, PP, PE,
                                                Theta_C1, Theta_P1,
                                                Theta_K13, Theta_K12)
    [BF, XX2out] = LowerSoilZone(XX2in, RG, Theta_K2Q)
    [SF, XX3out] = NashCasc(OF, NRes, XX3in, Theta_K3Q)
    QQ           = SF + BF

    # Compile outputs for time step
    Outputs[0] = QQ
    Outputs[1] = AE

    # Compile and get state values at end of time step
    OutStates[0] = XX1out
    OutStates[1] = XX2out
    for i in ResIdx:
        OutStates[i+2] = XX3out[i]

    # Compile internal fluxes for time step
    IntFluxes[0] = OF; IntFluxes[1] = PX; IntFluxes[2] = IF
    IntFluxes[3] = RG; IntFluxes[4] = BF; IntFluxes[5] = SF

    return [Outputs, OutStates, IntFluxes]

def MainCatchModel(Pars, PPobs, PEobs, QQobs, NRes):
# def MainCatchModel(PPobs, PEobs, Theta_C1,
#                    Theta_P1, Theta_K12, Theta_K13,
#                    Theta_K2Q, Theta_K3Q, NRes):
    
    NTime = len(PPobs)              # Get length of time vector
    Time  = np.arange(0, NTime, 1)  # Set up time vector (of integer values)

    # Pack parameter values in Par array
    # Npar = 6; Pars = np.zeros(Npar)
    # Pars[0] = Theta_C1;  Pars[1] = Theta_P1;  Pars[2] = Theta_K12
    # Pars[3] = Theta_K13; Pars[4] = Theta_K2Q; Pars[5] = Theta_K3Q

    Theta_K2Q = Pars[4]

    # Initialize model state arrays
    InStates = np.zeros(NRes+2); OutStates = InStates.copy()
    InStates[1] = QQobs[0]/Theta_K2Q

    # Initialize arrays
    Inputs = np.zeros(2); Outputs = np.zeros(2); IntFluxes = np.zeros(6)
    QQsim  = np.zeros(NTime); AEsim = np.zeros(NTime)

    OF = np.zeros(NTime); PX = np.zeros(NTime); IF = np.zeros(NTime)
    RG = np.zeros(NTime); BF = np.zeros(NTime); SF = np.zeros(NTime)
    XX1 = np.zeros(NTime+1); XX2 = np.zeros(NTime+1)
    XX3 = np.zeros([NTime+1, NRes])

    for t in Time:
        Inputs[0] = PPobs[t]; Inputs[1] = PEobs[t]
        [Outputs, OutStates, IntFluxes] = CatchModel(Inputs, InStates, Pars, NRes)

        QQsim[t] = Outputs[0].copy(); AEsim[t] = Outputs[1].copy()
        OF[t] = IntFluxes[0]; PX[t] = IntFluxes[1]; IF[t] = IntFluxes[2]
        RG[t] = IntFluxes[3]; BF[t] = IntFluxes[4]; SF[t] = IntFluxes[5]
        XX1[t+1] = OutStates[0]; XX2[t+1] = OutStates[1]
        ResIdx = np.arange(0, NRes, 1)
        for i in ResIdx:
            XX3[t+1, i] = OutStates[i+2]
        InStates = OutStates
    # use the below commented line when evaluating model state variables
    # return [QQsim, AEsim, XX1, XX2, XX3]

    SpinUp = 90                                         # SpinUp period to stabilize variations

    Qs   = QQsim[SpinUp:NTime].copy()                   # remove spin-up period
    Qo   = QQobs[SpinUp:NTime].copy()                   # remove spin-up period
    MSE  = MSE_Fn(Qs, Qo)                               # Compute MSE
    NSE  = NSE_Fn(Qs, Qo)                               # Compute NSE
    NSEL = NSE_Fn(np.log(Qs), np.log(Qo))               # Compute NSEL
    [KGEss, KGE, alpha, beta, rho] = KGE_Fn(Qs, Qo)     # Compute KGE & components

    return [QQsim, AEsim, MSE,NSE,NSEL,KGEss,KGE,alpha,beta,rho]

#====

#==== Compute Mean Square Error (MSE) ========================================

def MSE_Fn(Qs,Qo):                  # Define MSE_Fn function subprograms
    MSE = np.mean((Qs-Qo)**2)       # Compute MSE
    return MSE

#==== Compute NSE Metric

def NSE_Fn(Qs, Qo):                 # Define NSE_Fn function subprograms
    MSE = MSE_Fn(Qs,Qo)             # Compute MSE
    NSE = 1 - (MSE/np.var(Qo))      # Compute NSE
    return NSE

#==== Compute KGE Metric =====================================================

def KGE_Fn(Qs,Qo):                  # Define KGE_Fn function subprogram
    alpha = np.std(Qs)/np.std(Qo)                               # compute alpha
    beta  = np.mean(Qs)/np.mean(Qo)                             # compute beta
    rho   = np.corrcoef(Qs,Qo)[0,1]                             # compute rho
    KGE   = 1 - np.sqrt((1-alpha)**2 + (1-beta)**2 + (1-rho)**2)# compute KGE
    KGEss = (KGE - (1-np.sqrt(2)))/np.sqrt(2)                   # compute KGEss
    return [KGEss, KGE, alpha, beta, rho]   # Return compute quantities

#==== Compute all of the performance metrics (calls metric functions) ========

def PerfMetrics(QQsim, QQobs, SpinUp, iPrint):
    NTime = len(QQobs)
    Qs   = QQsim[SpinUp:NTime].copy()                   # remove spin-up period
    Qo   = QQobs[SpinUp:NTime].copy()                   # remove spin-up period
    MSE  = MSE_Fn(Qs, Qo)                               # Compute MSE
    NSE  = NSE_Fn(Qs, Qo)                               # Compute NSE
    NSEL = NSE_Fn(np.log(Qs), np.log(Qo))               # Compute NSEL
    [KGEss, KGE, alpha, beta, rho] = KGE_Fn(Qs, Qo)     # Compute KGE & components

    # Create lables for printing
    MSELab = f'MSE = {MSE:.2f}'; NSELab = f'NSE = {NSE:.2f}'
    KGEssLab = f'KGEss = {KGEss:.2f}'; RhoLab = f'Rho = {rho:.2f}'
    AlphaLab = f'Alpha = {alpha:.2f}'; BetaLab = f'Beta = {beta:.2f}'
    NSELLab = f'NSEL = {NSEL:.2f}'

    if iPrint == 1: # print following lines only if the iPrint Flag is set to 1
        print()
        print('================================================================')
        print(MSELab, NSELab, KGEssLab, AlphaLab, BetaLab, RhoLab, NSELLab)
        print('================================================================')

    return [MSE,NSE,NSEL,KGEss,KGE,alpha,beta,rho]

#==== Create Diagnostic Plots ===========================================================

def DiagnosticPlots(fignum,Time,QQsim,QQobs):
    
    MuQQsim = np.mean(QQsim);                                    # Compute mean of QQsim
    MuQQobs = np.mean(QQobs);                                    # Compute mean of QQobs
    SigQQsim = np.std(QQsim);                                    # Compute std  of QQsim
    SigQQobs = np.std(QQobs);                                    # Compute std  of QQobs
    rho     = np.corrcoef(QQsim,QQobs)[0, 1];                    # Compute Rho
    rho_srt = np.corrcoef(np.sort(QQsim),np.sort(QQobs))[0, 1];  # Compute Rho of sorted
    Lab1 = f'Mu_sim = {MuQQsim:.2f}, Sig_sim = {SigQQsim:.2f}'   # Create Labels
    Lab2 = f'Mu_obs = {MuQQobs:.2f}, Sig_obs = {SigQQobs:.2f}'   # Create Labels
 
    QQsimL    = np.log(QQsim);                                   # Create Log(QQsim)
    QQobsL    = np.log(QQobs);                                   # Create Log(QQobs)
    MuQQsimL  = np.mean(QQsimL);                                 # Compute mean of QQsimL
    MuQQobsL  = np.mean(QQobsL);                                 # Compute mean of QQobsL
    SigQQsimL = np.std(QQsimL);                                  # Compute std  of QQsimL
    SigQQobsL = np.std(QQobsL);                                  # Compute std  of QQobsL
    rhoL      = np.corrcoef(QQsimL,QQobsL)[0, 1];                   # Compute Rho
    rho_srtL  = np.corrcoef(np.sort(QQsimL),np.sort(QQobsL))[0,1];  # Compute Rho of sorted
    Lab1L = f'Mu_simL = {MuQQsimL:.2f}, Sig_simL = {SigQQsimL:.2f}' # Create Labels
    Lab2L = f'Mu_obsL = {MuQQobsL:.2f}, Sig_obsL = {SigQQobsL:.2f}' # Create Labels

    # First Row of Plots
    plt.figure(num = fignum);                                # Set figure number
    fig_width = 14; fig_height = 4;                          # Set figure width and height
    fig, ax1 = plt.subplots(nrows=1,ncols=1,\
                            figsize=(fig_width,fig_height)); # Create figure
    Period_Start = Time[366]; Period_End = Time[730];           # Set plot time period
    # Time Series Plot
    ax1.set_xlim(Period_Start,Period_End);                   # Fix xaxis limits
    ax1.set_xlabel('Time (Days)');                           # Set the x-axis label
    ax1.set_ylabel('QQ (mm)');                               # Set the x-axis label
    ax1.set_title('CatchModel: Time Series Plot');           # Add plot title
    ax1.grid('k--');                                         # Turn on grid lines
    B = [0,1.5*np.max(QQobs)];                               # Get the y-axis limits 
    ax1.set_ylim(B);                                         # Set the y-axis limits
    color1 = 'r.'; color2 = 'b';                             # Choose line color
    ax1.plot(Time,QQobs,color1,label="QQ observed");         # Plots the Observed Flows
    ax1.plot(Time,QQsim,color2,label="QQ simulated");        # Plots the Simulated Flows
    ax1.legend(loc=2);
    
    # Second Row of Plots
    fignum = fignum + 1;
    plt.figure(num = fignum);                                # Set figure number
    fig_width = 14; fig_height = 6.5;                        # Set figure width and height
    fig, ax2 = plt.subplots(nrows=1,ncols=2,\
                            figsize=(fig_width,fig_height)); # Create figure
    # Scatterplot
    ax2[0].set_xlabel('QQobs',color='r');                    # Set the x-axis label
    ax2[0].set_ylabel('QQsim',color='b');                    # Set the y-axis label
    ax2[0].grid('k--');                                      # Turn on grid lines
    B = [0,1.25*np.max(QQobs)];                              # Get the axis limits 
    ax2[0].set_xlim(B);                                      # Set the x-axis limits
    ax2[0].set_ylim(B);                                      # Set the y-axis limits
    ax2[0].plot(QQobs,QQsim,'k.');                           # Plots a scatterplot
    ax2[0].plot(B,B,'k:');                                   # Plot 1:1 line
    ax2[0].set_title(f'Rho = {rho:.2f}');                    # Add plot title
    ax2[0].plot(B,[MuQQsim,MuQQsim],'b:',label = Lab1);
    ax2[0].tick_params(axis='y',labelcolor = 'b');           # Set the yaxis tick color
    ax2[0].plot([MuQQobs,MuQQobs],B,'r:',label = Lab2);
    ax2[0].tick_params(axis='x',labelcolor = 'r');           # Set the xaxis tick color
    ax2[0].legend(loc=2);
    # Q-Q Plot
    ax2[1].set_xlabel('sort(QQobs)',color='r');              # Set the x-axis label
    ax2[1].set_ylabel('sort(QQsim)',color='b');              # Set the y-axis label
    ax2[1].grid('k--');                                      # Turn on grid lines
    B = [0,1.25*np.max(QQobs)];                              # Get the axis limits 
    ax2[1].set_xlim(B);                                      # Set the x-axis limits
    ax2[1].set_ylim(B);                                      # Set the y-axis limits
    ax2[1].plot(np.sort(QQobs),np.sort(QQsim),'k.');         # Plots a Q-Q Plot
    ax2[1].plot(B,B,'k:');                                   # Plot 1:1 line
    ax2[1].set_title(f'Rho = {rho_srt:.2f}');                # Add plot title
    #ax2[1].plot(B,[MuQQsim,MuQQsim],'b:',label = Lab1);
    ax2[1].tick_params(axis='y',labelcolor = 'b');           # Set the yaxis tick color
    #ax2[1].plot([MuQQobs,MuQQobs],B,'r:',label=Lab2);
    ax2[1].tick_params(axis='x',labelcolor = 'r');           # Set the xaxis tick color
    #ax2[1].legend(loc=2);

    #Third Row of Plots
    fignum = fignum + 1;    
    plt.figure(num = fignum);                               # Set figure number
    fig_width = 14; fig_height = 4;                         # Set figure width and height
    fig, ax3 = plt.subplots(nrows=1,ncols=1,\
                            figsize=(fig_width,fig_height)); # Create figure
    # Time Series Log Plot
    ax3.set_xlim(Period_Start,Period_End);                # Fix xaxis limits
    ax3.set_xlabel('Time (Days)');                        # Set the x-axis label
    ax3.set_ylabel('Log(QQ)');                            # Set the x-axis label
    ax3.set_title('CatchModel: Time Series Plot');        # Add plot title
    ax3.grid('k--');                                      # Turn on grid lines
    B = [0.1,1.5*np.max(QQobs)];                          # Get the y-axis limits 
    ax3.set_ylim(B);                                      # Set the y-axis limits
    color1 = 'r.'; color2 = 'b';                          # Choose line color
    ax3.semilogy(Time,QQobs,color1,label="QQ observed");  # Plots the Observed Flows
    ax3.semilogy(Time,QQsim,color2,label="QQ simulated"); # Plots the Simulated Flows
    ax3.legend(loc=2);

    # Fourth Row of Plots
    fignum = fignum + 1;
    plt.figure(num = fignum);                                # Set figure number
    fig_width = 14; fig_height = 6.5;                        # Set figure width and height
    fig, ax4 = plt.subplots(nrows=1,ncols=2,\
                            figsize=(fig_width,fig_height)); # Create figure
    # Scatterplot
    ax4[0].set_xlabel('Log(QQobs)',color='r');               # Set the x-axis label
    ax4[0].set_ylabel('Log(QQsim)',color='b');               # Set the y-axis label
    ax4[0].grid('k--');                                      # Turn on grid lines
    B = [0.1,1.25*np.max(QQobs)];                            # Get the axis limits 
    ax4[0].set_xlim(B);                                      # Set the x-axis limits
    ax4[0].set_ylim(B);                                      # Set the y-axis limits
    ax4[0].loglog(QQobs,QQsim,'k.');                         # Plots a scatterplot
    ax4[0].loglog(B,B,'k:');                                 # Plot 1:1 line
    ax4[0].set_title(f'Rho = {rhoL:.2f}');                   # Add plot title
    ax4[0].loglog(B,[MuQQsim,MuQQsim],'b:',label = Lab1L);
    ax4[0].tick_params(axis='y',labelcolor = 'b');           # Set the yaxis tick color
    ax4[0].loglog([MuQQobs,MuQQobs],B,'r:',label = Lab2L);
    ax4[0].tick_params(axis='x',labelcolor = 'r');           # Set the xaxis tick color
    ax4[0].legend(loc=2);
    # Q-Q Plot
    ax4[1].set_xlabel('sort(Log(QQobs)',color='r');          # Set the x-axis label
    ax4[1].set_ylabel('sort(Log(QQsim))',color='b');         # Set the y-axis label
    ax4[1].grid('k--');                                      # Turn on grid lines
    B = [0.1,1.25*np.max(QQobs)];                            # Get the axis limits 
    ax4[1].set_xlim(B);                                      # Set the x-axis limits
    ax4[1].set_ylim(B);                                      # Set the y-axis limits
    ax4[1].loglog(np.sort(QQobs),np.sort(QQsim),'k.');       # Plots a Q-Q Plot
    ax4[1].loglog(B,B,'k:');                                 # Plot 1:1 line
    ax4[1].set_title(f'Rho = {rho_srtL:.2f}');               # Add plot title
    #ax4[1].plot(B,[MuQQsim,MuQQsim],'b:',label = Lab1);
    ax4[1].tick_params(axis='y',labelcolor = 'b');           # Set the yaxis tick color
    #ax4[1].plot([MuQQobs,MuQQobs],B,'r:',label=Lab2);
    ax4[1].tick_params(axis='x',labelcolor = 'r');           # Set the xaxis tick color
    #ax4[1].legend(loc=2);
    
    return
# %%
