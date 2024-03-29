import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def obj_func(Pars,PPobs,PEobs,QQobs,NRes):
    
    NTime = len(PPobs);             # Get length of Time vector
    Time = np.arange(0,NTime,1);    # Set up Time vector (of integer values) 
    
    # Pack Parameter Values into Par Array
    # Npar = 6; Pars = np.zeros(Npar);
    Theta_C1 = Pars[0]; Theta_P1 = Pars[1]; Theta_K12 = Pars[2];
    Theta_K13 = Pars[3]; Theta_K2Q = 10**Pars[4]; Theta_K3Q = Pars[5];
    
    # Initialize Model State Arrays
    InStates = np.zeros(NRes+2); OutStates = InStates.copy();
    InStates[1] = QQobs[0]/Theta_K2Q;

    # Initialize Arrays
    Inputs = np.zeros(2); Outputs = np.zeros(2); IntFluxes = np.zeros(6); 
    QQsim  = np.zeros(NTime); AEsim  = np.zeros(NTime); 
    OF = np.zeros(NTime); PX = np.zeros(NTime); IF = np.zeros(NTime);
    RG = np.zeros(NTime); BF = np.zeros(NTime); SF = np.zeros(NTime);
    XX1 = np.zeros(NTime+1); XX2 = np.zeros(NTime+1);
    XX3 = np.zeros([NTime+1,NRes]);
    
    for t in Time: # Run CatchModel loop on time
        Inputs[0] = PPobs[t]; Inputs[1] = PEobs[t];
        [Outputs, OutStates, IntFluxes] = CatchModel(Inputs,InStates,Pars,NRes);
        
        QQsim[t] = Outputs[0].copy(); AEsim[t] = Outputs[1].copy();
        OF[t] = IntFluxes[0]; PX[t] = IntFluxes[1]; IF[t] = IntFluxes[2];
        RG[t] = IntFluxes[3]; BF[t] = IntFluxes[4]; SF[t] = IntFluxes[5];
        XX1[t+1] = OutStates[0]; XX2[t+1] = OutStates[1];
        ResIdx = np.arange(0,NRes,1);
        for i in ResIdx:
            XX3[t+1,i] = OutStates[i+2];  
        InStates = OutStates;
    
    #return [QQsim,AEsim]

    #def PerfMetrics(QQsim,QQobs,SpinUp,iPrint): # Compute Perf. Metrics on Desired Period 
    
    SpinUp = 90

    Qs   = QQsim[SpinUp:NTime].copy();                     # Remove SpinUp period
    Qo   = QQobs[SpinUp:NTime].copy();                     # Remove SpinUp period
    MSE  = MSE_Fn(Qs,Qo);                                  # Compute MSE
    NSE  = NSE_Fn(Qs,Qo);                                  # Compute NSE
    NSEL = NSE_Fn(np.log(Qs),np.log(Qo));                  # Compute NSEL
    [KGEss, KGE, alpha, beta, rho] = KGE_Fn(Qs,Qo);        # Compute KGE & components
    
    # Create labels for Printing
    MSELab = f'MSE = {MSE:.2f}'; NSELab = f'NSE = {NSE:.2f}'; 
    KGEssLab = f'KGEss = {KGEss:.2f}'; RhoLab = f'Rho = {rho:.2f}';
    AlpLab = f'Alpha = {alpha:.2f}'; BetLab = f'Beta = {beta:.2f}'; 
    NSELLab = f'NSEL = {NSEL:.2f}';
    
    #print(KGEss)

    return 1/KGEss

#==== Compute MSE Metric ================================================================

def MSE_Fn(Qs,Qo):                        # Define MSE_Fn function subprogram
    MSE = np.mean((Qs-Qo)**2);            # Compute MSE
    return MSE

#==== Compute NSE Metric ================================================================

def NSE_Fn(Qs,Qo):                          # Define NSE_Fn function subprogram
    MSE = MSE_Fn(Qs,Qo);                    # Compute MSE
    NSE = 1 - (MSE/np.var(Qo));             # Compute NSE
    return NSE

#==== Compute KGE Metric ================================================================

def KGE_Fn(Qs,Qo):                          # Define KGE_Fn function subprogram    
    beta  = np.mean(Qs)/np.mean(Qo);                              # Compute Beta
    alpha = np.std(Qs)/np.std(Qo);                                # Compute Alpha
    rho   = np.corrcoef(Qs,Qo)[0,1];                              # Compute Rho
    KGE   = 1 - np.sqrt((1-alpha)**2 + (1-beta)**2 + (1-rho)**2); # Compute KGE
    KGEss = (KGE - (1-np.sqrt(2)))/np.sqrt(2);                    # Compute KGEss    
    return [KGEss, KGE, alpha, beta, rho];  # Return computed quantities

#==== Catchment Model (for One Time Step) ===============================================

def CatchModel(Inputs,InStates,Pars,NRes): # Runs the Catchment Model for One Time Step
    
    # Get Input Values for Time Step
    PP = Inputs[0]; 
    PE = Inputs[1];
    
    # Get State Values at Begining of Time Step
    XX1in  = InStates[0];
    XX2in  = InStates[1];
    XX3in = np.zeros(NRes); 
    ResIdx = np.arange(0,NRes,1);
    for i in ResIdx:
        XX3in[i] = InStates[i+2];
        
    #Get Par Values
    Theta_C1  = Pars[0]; 
    Theta_P1  = Pars[1]; 
    Theta_K12 = Pars[2]; 
    Theta_K13 = Pars[3];
    Theta_K2Q = 10**Pars[4];
    Theta_K3Q = Pars[5];
    
    # Initialize Output Array
    Outputs = np.zeros(2);
    OutStates = np.zeros(NRes+2);
    IntFluxes = np.zeros(6);
    
    # Run Model Components
    [OF, PX, IF, AE, RG, XX1out] = UpperSoilZone(XX1in, PP, PE, \
                                    Theta_C1, Theta_P1, Theta_K13, Theta_K12);
    [BF, XX2out] = LowerSoilZone(XX2in, RG, Theta_K2Q);
    [SF, XX3out] = NashCasc(OF, NRes, XX3in, Theta_K3Q);
    QQ           = SF + BF;
    
    # Compile Outputs for Time Step
    Outputs[0] = QQ; Outputs[1] = AE;
    
    # Compile Get State Values at End of Time Step
    OutStates[0] = XX1out; OutStates[1] = XX2out;
    for i in ResIdx:
        OutStates[i+2] = XX3out[i];
        
    # Compile Internal Fluxes for Time Step    
    IntFluxes[0] = OF; IntFluxes[1] = PX; IntFluxes[2] = IF;
    IntFluxes[3] = RG; IntFluxes[4] = BF; IntFluxes[5] = SF;
    
    return [Outputs, OutStates, IntFluxes]

#==== Upper Soil Zone Module (for one time step) ========================================

def UpperSoilZone(XX1in, PP, PE, Theta_C1, Theta_P1, Theta_K13, Theta_K12):
    
    PX     = np.maximum(0, PP + XX1in - Theta_C1);
    IF     = Theta_K13*XX1in;
    OF     = PX + IF;
    XX1a   = XX1in - IF;
    AE     = np.minimum(XX1a, PE*(XX1a/Theta_C1)**Theta_P1);
    XX1b   = XX1a - AE;
    RG     = Theta_K12*XX1b;
    XX1c   = XX1b - RG;
    XX1out = XX1c + PP - PX;
    
    return [OF, PX, IF, AE, RG, XX1out]

#==== Lower Soil Zone Module (for one time step) ========================================

def LowerSoilZone(XX2in, RG, Theta_K2Q):
    [BF, XX2out] = LinRes(XX2in, RG, Theta_K2Q);
    return [BF, XX2out]

#==== Channel Routing Module (for one time step) ========================================

def NashCasc(OF, NRes, XX3in, Theta_K3Q): # Define NashCasc function subprogram
    
    ResIdx = np.arange(0,NRes,1);         # Set Loop vector on tanks (integer values) 
    UU     = OF;                          # UU is set to be input to first tank
    XX3out = np.zeros(NRes);              # Initialize array to recieve XX3out
    YY     = np.zeros(NRes);              # Initialize array to recieve YY (individual tank outflows)
    
    for i in ResIdx:                # Begin loop on tanks in series
        [YY[i], XX3out[i]] = LinRes(XX3in[i], UU, Theta_K3Q); # Call LinRes subprogram
        UU = YY[i];                       # Set UU (inflow to next tank) to be outflow from previous tank
    SF = UU;                              # Get output from last tank
    
    return [SF, XX3out];                  # Return computed quantities

#==== Linear Reservoir Module (for one time step) =======================================

def LinRes(Xin, U, Kpar):                 # Define LinRes function subprogram  
    
    Y = Kpar*Xin;                         # Compute QQ
    Xout = Xin - Y + U;                   # Update state variable
    
    return [Y, Xout]                      # Return compute quantities