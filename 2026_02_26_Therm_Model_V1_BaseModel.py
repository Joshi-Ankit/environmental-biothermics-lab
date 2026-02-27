# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 12:41:15 2025

@author: Ankit Joshi, Gisel Guzman translated with ChatGPT from his matlab implementation

This thermoregulation model is based on J.A.J Stolwijk's model published in 1971.
Material properties and other parameters were same as per stolwijk's-1971 model but following things has been modified

(1) Implementataion of clothing
(2) Updated heat transfer coefficient values as per Ichihara-1997, and Takahashi-2021 (JOS3 model)
(3) Updated set-point temperatures as per Takahashi-2021 (JOS3 model)
(4) Updated control coefficients as per Takahashi-2021 (JOS3 model)
(5) Updated the blood flow
Convective and radiative heat transfer coefficients are improvised based on latest findings

% % Constants
% k_air = 0.026;      % thermal conductivity of air [W/mÂ·K]
% nu_air = 1.5e-5;     % kinematic viscosity of air [mÂ²/s]
% D = 0.173;            % characteristic body diameter [m]

"""

# stolwijk2024.py
# Single-file, no-classes port of the "Stolwijk 2024" MATLAB script.
# Dependencies: numpy, pandas, matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
from pathlib import Path

from IPython import get_ipython

# =============================================================================
# Shivering model selection
# =============================================================================
active_model = 'stolwijk' #"stolwijk"     # options: "stolwijk" or "fiala"
skinBF_model = 'stolwijk' #"stolwijk" 
ar_ve_network = 'no'

#active_model = 'fiala' #"stolwijk"     # options: "stolwijk" or "fiala"
#skinBF_model = 'fiala' #"stolwijk" 
ar_ve_network = 'yes'


# --- Optional: clear figures when running interactively ---
plt.close('all')

rcParams['legend.frameon']= False 
rcParams['legend.markerscale']=2.
rcParams['legend.fontsize']=7.
rcParams['axes.edgecolor']='0.8'
rcParams['axes.labelcolor']='0.15'
rcParams['axes.linewidth']='0.8'
rcParams['axes.labelsize']=7
rcParams['axes.titlesize']=8
rcParams[u'text.color']= u'.15'
rcParams[u'xtick.direction']= u'in'
rcParams[u'xtick.major.width']= 0.5
rcParams[u'xtick.labelsize']= 7
rcParams[u'ytick.labelsize']= 7
rcParams[u'ytick.color']=u'.15'
rcParams[u'ytick.direction']=u'in'
rcParams[ u'font.sans-serif']=[u'Arial',
                               u'Liberation Sans',
                               u'Bitstream Vera Sans',
                               u'sans-serif']


# =============================================================================
# Setting workdir and metadata
# =============================================================================

# from pathlib import Path

# =============================================================================
# Paths & I/O
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
name_simulation = '2022_Smallcombe'

# Default: look for the input Excel next to this script.
# You can override by setting INPUT_XLSX (absolute path) below.
#INPUT_XLSX = BASE_DIR / '1998_Brajkovic_Input.xlsx'
#INPUT_XLSX = BASE_DIR / '1992_Vallerand_Input.xlsx'
INPUT_XLSX = BASE_DIR / '2022_Smallcombe_input.xlsx'
if not INPUT_XLSX.exists():
    raise FileNotFoundError(f"Input Excel not found: {INPUT_XLSX}")

InputData = pd.read_excel(INPUT_XLSX, index_col=None)


# =============================================================================
# The model
# =============================================================================
# -------------------------------------------------
# -------------Input parameters--------------------
# -------------------------------------------------
dTime = 1.0             # [s]
start_time = 0.0        # [s]
end_time = (400)*60   # [s]
outputInterval = 600.0   # [s]

weight = 74.43  # kg (unused here but kept)
height = 1.81   # m  (unused here but kept)
age = 20        # years (unused here but kept)



# -------------------------------------------------
# -------------Other parameters--------------------
# -------------------------------------------------
P_atm = 101325.0                 # [Pa]
L_c = 16.5 / (P_atm / 101325.0)  # [K/kPa] (original scalar; later recomputed) Lewis coefficient (script scales by 1/1000 to mix with Pa).

# -------------------------------------------------
# ----------Thermal properties---------------------
# -------------------------------------------------
# Node order per segment (0-based in Python):
# core = 4*i+0, muscle = 4*i+1, fat = 4*i+2, skin = 4*i+3  for i in [0..5]
# Central blood pool is index 24
# Heat capacitance [J/K] (length 25) in original.ref kcal.C
C = np.array([
    9288.48,  1380.72,  920.48,  1004.16,   # Head
    41086.88, 67571.6,  17782.0, 5062.64,   # Trunk
    5899.44,  12719.36, 2426.72, 1799.12,   # Arms
    585.76,   251.04,   376.56,  711.28,    # Hands
    17740.16, 38367.28, 5983.12, 4518.72,   # Legs
    962.32,   251.04,   543.92,  920.48,    # Feet
    9414.00                                    # Central blood
], dtype=float)

# Basal metabolic heat generation [W] (length 25)
Q_bm = np.array([
    14.93, 0.12, 0.13, 0.09,     # Head
    52.78, 5.82, 2.48, 0.47,     # Trunk
    0.81,  1.10, 0.20, 0.15,     # Arms
    0.09,  0.23, 0.03, 0.06,     # Hands
    2.59,  3.33, 0.50, 0.37,     # Legs
    0.15,  0.02, 0.05, 0.08,     # Feet
    0.00                         # Central blood
], dtype=float)

# Basal effective blood flow [l/h] (length 25)
BFB = np.array([
    45.00,  0.12, 0.13, 1.44,    # Head
    210.00, 6.00, 2.56, 2.10,    # Trunk
    0.84,   1.14, 0.20, 0.50,    # Arms
    0.10,   0.24, 0.04, 2.00,    # Hands
    2.69,   3.43, 0.52, 2.85,    # Legs
    0.16,   0.02, 0.05, 3.00,    # Feet
    0.00                         # Central blood
], dtype=float)

# Thermal conductance between j and j+1 [W/K] (length 25)
TC = np.array([
    1.605,  13.258, 16.049, 0.000,  # Head
    1.593,  5.524,  23.027, 0.000,  # Trunk
    1.396,  10.351, 30.471, 0.000,  # Arms
    6.397,  11.223, 11.514, 0.000,  # Hands
    10.467, 14.421, 74.432, 0.000,  # Legs
    16.282, 20.585, 16.398, 0.000,  # Feet
    0.000                            # Central blood
], dtype=float)

# T_set as per JOS3 [K] (length 25)
T_set = np.array([
    310.65, 308.69, 308.58, 307.65,  # Head
    310.18, 309.57, 308.87, 307.93,  # Trunk
    309.29, 307.86, 307.69, 307.34,  # Arms
    308.20, 308.17, 307.67, 307.59,  # Hands
    309.81, 309.29, 308.46, 307.25,  # Legs
    307.89, 307.78, 307.46, 307.39,  # Feet
    310.49                           # Central blood
], dtype=float)

# -------------------------------------------------
# ------Model constants & distribution factors------
# -------------------------------------------------
# segment order: Head, Trunk, Arms, Hands, Legs, Feet
SA    = np.array([0.133, 0.680, 0.254, 0.095, 0.597, 0.130])   # [m^2] Surface area of body segments
DF_work = np.array([0.000, 0.300, 0.080, 0.010, 0.600, 0.010]) # [-] Distributuion factor of total work done by muscles in given segment 
DF_TR   = np.array([0.0695,0.4935,0.0686,0.1845,0.1505,0.0334])# [-] Distributuion factor of skin thermoreceptor in given segment
DF_VD   = np.array([0.1320,0.3220,0.0950,0.1210,0.2300,0.1000])# [-] Distributuion factor of VD in given segment
DF_VC   = np.array([0.0500,0.1500,0.0500,0.3500,0.0500,0.3500])# [-] Distributuion factor of VC in given segment
DF_SW   = np.array([0.0810,0.4810,0.1540,0.0310,0.2180,0.0350])# [-] Distributuion factor of SW in given segment (fractional distribution of sweat glands over different body segments,  described in SM Joshi 2024)
DF_SH   = np.array([0.0200,0.8500,0.0500,0.0000,0.0700,0.0000])# [-] Distributuion factor of shivering heat generation by muscles in given segment

# --- Fiala shivering parameters (setpoints must match your control model) ---
Tskm_set = 34.5+273.15;   # [°C] mean skin setpoint (choose your model's value)
Thy_set  = 37.1+273.15;   # [°C] hypothalamus/core setpoint (choose your model's value)
T_art = 310.05 * np.ones(6)
hx_ccx = np.array([0.000, 0.000, 0.485, 0.7600, 0.635, 0.99], dtype=float)

# -------------------------------------------------
# ------------Initialization------------------------
# -------------------------------------------------
numTimeSteps = int(round((end_time - start_time) / dTime)) + 1 #number of time steps
outputEverySteps = max(1, int(round(outputInterval / dTime)))  # sample every N solver steps
resultIndex = 0 #Counter for result storageH
T = T_set.copy()               # Start at set point [K] (temperatures at time 0)
timeArray = np.zeros(numTimeSteps)  # time [s]
outputEverySteps = max(1, int(round(outputInterval / dTime)))  # sample every N solver steps
SA_total = SA.sum() #Total surface area in [m^2]

time_input = InputData['time_min']*60 #time from excel in [s]


#Python initialization setting up space----------------------------------------
# Pre-size output arrays (max number of rows we’ll store, they are time series)
numOutputRows = int(np.floor((end_time - start_time) / outputInterval)) + 1
TimeStamp = np.zeros(numOutputRows)  # [s]

T_output = np.zeros((numOutputRows, 25)) #Temperature at each node
w_sk_output = np.zeros((numOutputRows, 1))
Q_storage_output = np.zeros((numOutputRows, 1))
T_ms_output = np.zeros((numOutputRows, 1))
BF_skin_output  = np.zeros((numOutputRows, 1))
CO_output = np.zeros((numOutputRows, 1))
Skin_wettedness_output = np.zeros((numOutputRows, 6))
BF_output = np.zeros((numOutputRows, 24))
Total_sweat_secretion_output = np.zeros((numOutputRows, 1))
Total_sweat_evaporation_output = np.zeros((numOutputRows, 1))

Qsk_latent_total_output = np.zeros((numOutputRows, 1))
Qsk_sensible_total_output = np.zeros((numOutputRows, 1))
SHL_sk_output = np.zeros((numOutputRows, 6))
LHL_sk_output = np.zeros((numOutputRows, 6))
THL_sk_output = np.zeros((numOutputRows, 6))
RES_output = np.zeros((numOutputRows, 1))

Tair_output = np.zeros((numOutputRows, 1))
MRT_output = np.zeros((numOutputRows, 1))
RH_air_output = np.zeros((numOutputRows, 1))
To_output = np.zeros((numOutputRows, 6))
To_mean_output = np.zeros((numOutputRows, 1))
v_air_output = np.zeros((numOutputRows, 1))
met_output = np.zeros((numOutputRows, 1))
Shivering_output = np.zeros((numOutputRows, 1))
VD_output = np.zeros((numOutputRows, 1))
VC_output = np.zeros((numOutputRows, 1))
SW_output = np.zeros((numOutputRows, 1))
fx_output = np.zeros((numOutputRows, 1))

Tskm_prev = None

# -------------------------------------------------
# ----Time loop (Euler forward integration)--------
# -------------------------------------------------
for stepIndex in range(numTimeSteps):
    currentTime = start_time + stepIndex * dTime #Time step for the current loop [s]
    #print(f"Current Time: {currentTime:.0f} s") #Display the time in console [s]
    
    # --- Find last defined time <= current time ---
    #this is to produce a stepwise assignment on environmental data (no interpolation)
    # Note: MATLAB find(..., 'last') returns 1-based index; Python uses 0-based.
    idt = int(np.searchsorted(time_input, currentTime, side='right') - 1)
    if idt < 0:
        idt = 0
    elif idt >= len(time_input):
        idt = len(time_input) - 1
        
    #Notice If somehow I want the closest time to the measurement I could use
    #idt = int(np.nanargmin(np.abs(time_input - currentTime)))
    
    
    # ---------- Environmental | Activity | position schedule:  ------------------------
    
    T_air  = InputData.iloc[idt, 1]
    MRT    = InputData.iloc[idt, 2]
    v_air  = InputData.iloc[idt, 3]
    RH_air = InputData.iloc[idt, 4]
    met    = InputData.iloc[idt, 5]
    
    Rc_cl = np.zeros(6)
    Rc_cl[0] = InputData.iloc[idt, 6]
    Rc_cl[1] = InputData.iloc[idt, 7]
    Rc_cl[2] = InputData.iloc[idt, 8]
    Rc_cl[3] = InputData.iloc[idt, 9]
    Rc_cl[4] = InputData.iloc[idt,10]
    Rc_cl[5] = InputData.iloc[idt,11]
    
    Re_cl = np.zeros(6)
    Re_cl[0] = InputData.iloc[idt,12]
    Re_cl[1] = InputData.iloc[idt,13]
    Re_cl[2] = InputData.iloc[idt,14]
    Re_cl[3] = InputData.iloc[idt,15]
    Re_cl[4] = InputData.iloc[idt,16]
    Re_cl[5] = InputData.iloc[idt,17]
    posture = "standing" #Posture
                

    # ----------------- Processing environmental parameters -------------------
    T_air_K = T_air + 273.15 #Converting to kelvin
    MRT_K = MRT + 273.15     #Converting to kelvin

    # Saturation pressure of water vapor in air [Pa]
    Psat_air = np.exp(77.3450 + 0.0057 * T_air_K - 7235.0 / T_air_K) / (T_air_K ** 8.2) #[Pa] Saturation pressure of water vpour in air 
    P_air = RH_air * Psat_air                       # [Pa] Actual pressure of water vapour in air
    P_atm = 101325.0                                # [Pa] Atmospheric pressure (This is sea level pressure)
    L_c = (16.5 / (P_atm / 101325.0)) / 1000.0      # [-] Lewis coefficient (kPa^-1*K, used as scalar)
    
    #--------Identify skin node temperatures in the current time (K) ----------
    T_sk = np.zeros(6)
    for i in range(6):
        skin_idx = 4 * i + 3    #To identify the skin nodes [-]
        T_sk[i] = T[skin_idx]   #Skin temperature [deg C]

    # -------------------------------------------------
    # -----setting parameters to heat transfer to environment-----------
    # -------------------------------------------------
    #--------------------- Heat transfer coefficients -------------------------
    if v_air >= 0.2: # Forced convection, standing
        #BY default:
        # hc_a = np.array([15.00, 13.67, 17.00, 20.00, 15.02, 15.10]) #[Head, Trunk, Arms, Hands, Legs, Feet]
        # hc_b = np.array([0.62,  0.57,  0.60,  0.60,  0.68,  0.62 ]) #[Head, Trunk, Arms, Hands, Legs, Feet]
        
        #For Smallcombe 2022:
        hc_a = np.array([3.20, 8.1, 11.09, 14.40, 11.10, 12.00])    #[Head, Trunk, Arms, Hands, Legs, Feet]
        hc_b = np.array([0.97,  0.62,  0.59,  0.55, 0.51,   0.49])  #[Head, Trunk, Arms, Hands, Legs, Feet]

        
        h_rad = np.array([4.89, 4.26, 4.49, 4.21, 5.08, 6.14])      #Radiation heat transfer coefficients for given body segments [W m^-2 K^-1]
        h_conv = hc_a * (v_air ** hc_b)                             #Forced convection heat transfer coefficient [W m^-2 K^-1]
        # print("Forced convection, standing")
    else: # Natural convection
        if (posture == "sitting") and (v_air < 0.2):
            h_conv = np.array([4.75, 2.49, 3.69, 2.06, 2.98, 2.62]) #Natural convection heat transfer coefficient [W m^-2 K^-1]
            h_rad  = np.array([4.96, 4.27, 4.59, 4.74, 4.44, 6.36]) #Radiation heat transfer coefficients for given body segments [W m^-2 K^-1]
            # print("Natural convection, sitting")
        elif (posture == "supine") and (v_air < 0.2):
            hc_a = np.array([1.11, 1.21, 1.49, 2.18, 0.67, 0.20]) 
            hc_b = np.array([0.35, 0.05, 0.70, 0.30, 0.49, 0.62]) #Radiation heat transfer coefficients for given body segments [W m^-2 K^-1]
            # Natural convection depends on |T_skin - T_air|
            h_conv = np.zeros(6)
            for i in range(6):
                skin_idx = 4 * i + 3
                h_conv[i] = hc_a[i] * (abs(T[skin_idx] - T_air_K) ** hc_b[i]) #Natural convection heat transfer coefficient [W m^-2 K^-1]
            h_rad = np.array([5.48, 3.46, 4.52, 4.12, 5.03, 6.09])
            # print("Natural convection, supine")
        elif (posture == "standing") and (v_air < 0.2):
            h_conv = np.array([4.48, 2.91, 3.59, 3.67, 2.53, 2.04]) #Natural convection heat transfer coefficient [W m^-2 K^-1]
            h_rad  = np.array([4.89, 4.26, 4.49, 4.21, 5.08, 6.14]) #Radiation heat transfer coefficients for given body segments [W m^-2 K^-1]
            # print("Natural convection, standing")
        else:
            # Fallback (shouldn't trigger in your schedule)
            raise ValueError(f"non-valid combination of {posture} and wind speed.")

    #----------------- Heat transfer through clothing -------------------------
    f_cl = 1.00 + 1.81 * Rc_cl                                    # clothing area factor: adjusting the overall heat transfer based on the ratio of the clothed surface area to the nude surface area. 
    h_total = 1.0 / (Rc_cl + 1.0 / (f_cl * (h_conv + h_rad)))     # [W m^-2 K^-1] Total heat transfer coefficient. Reciprocal of (Rc) clothing resistance and air layer resistance
    T_o = (h_conv * T_air_K + h_rad * MRT_K) / (h_conv + h_rad)   # [K] Operative temp 

    # -------------------------------------------------
    # ----------Control system signals-----------------
    # -------------------------------------------------
    
    # Error signals estimation
    WRM = np.zeros(25) #Initialize warm array [-]
    CLD = np.zeros(25) #Initialize cold array [-]
    
    ERR = T - T_set    #ERR signal offset from the setpoint temperature [degC]
    WRM = np.maximum(ERR, 0.0)  #Warm signal[-]
    CLD = np.abs(np.minimum(ERR, 0.0)) #Cold signal[-]

    # (Integrate pheripheral afferents) Integration of skin thermoreceptor signals
    # ----------------------------------------------------------------------------
    WRM_skin = np.zeros(6)
    CLD_skin = np.zeros(6)
    for i in range(6):
        skin_idx = 4 * i + 3 #To identify the skin nodes [-]
        WRM_skin[i] = WRM[skin_idx] * DF_TR[i] # Weighing based on skin receptors[-]
        CLD_skin[i] = CLD[skin_idx] * DF_TR[i] #Weighing based on skin receptors[-]
    WRMS = WRM_skin.sum() #Integration of skignal from all the zones [-]
    CLDS = CLD_skin.sum() #Integration of skignal from all the zones [-]

    # (from 1971 this part has elements from: Determination of efferent outflow AND Assign effector output)
    # #1971: A linear form of the weighted value of the skin error signal can be recovered by the use of WARMS - COLDS.
    # ----------------------------------------------------------------------------

    # Skin blood flow (This is and update from JOS3 control) -------------------
    # =============================================================================
    # THERMOREG CONTROLLER (SWITCHABLE): Stolwijk/JOS3 vs Fiala
    # Outputs (always defined): VD, VC, Shivering, SW
    # =============================================================================
    
    # ---- initialize outputs so they're always defined ----
    VD = 0.0
    VC = 0.0
    Shivering = 0.0
    SW = 0.0     # sweating drive (signal), not necessarily sweat rate
    
    # Fiala-native signals (for optional debugging)
    DL = 0.0
    CS = 0.0
    SH = 0.0
    
    Tskm_i = np.zeros(6)
    
    for i in range(6):   # MATLAB 1:6 -> Python range(6)
        Tskm_i[i] = T_sk[i] * (SA[i] / SA_total)
    
    Tskm = np.sum(Tskm_i)
    
    if active_model.lower() == "stolwijk":
        # -------------------------------------------------------------------------
        # STOLWIJK / JOS3-style controller signals (your given equations)
        # -------------------------------------------------------------------------
        #SH = 109.8* dTskm_dt_cool + 24.36 * (-ERR[0]) * CLDS
        #VD = max((100.5 * ERR[0]) + (6.4 * (WRMS - CLDS)), 0.0)      # vasodilation drive
        #VC = max((-10.8 * ERR[0]) + (-10.8 * (WRMS - CLDS)), 0.0)    # vasoconstriction drive
        # If you want Stolwijk shivering too (your earlier form)
        VD = max((117.0 * ERR[0]) + (7.5 * (WRMS - CLDS)), 0.0) #Stolwijk
        VC = (max(abs((-5.0 * ERR[0]) + ((WRMS - CLDS))), 0.0)) #Stolwijk
        SW = max((371.2 * ERR[0]) + (33.64 * (WRMS - CLDS)), 0.0)        # [W], JOS3 SW signal
        # (keep or remove depending on your current validated module)
        Shivering = max(0,24.36 * (-ERR[0]) * CLDS) #Shivering signal
        Shivering = float(np.clip(Shivering, 0.0, 350.0))  # optional clamp
    
        # If your sweating is already validated elsewhere, leave SW as 0 here.
        # Otherwise you can add your Stolwijk sweat signal mapping here.
    
    
    elif active_model.lower() == "fiala":
        # -------------------------------------------------------------------------
        # FIALA whole-body controller (SH, CS, DL, SW)
        # Requires: Tskm, Thy, Tskm_set, Thy_set, dTime (sec or min)
        # -------------------------------------------------------------------------
        # initialize previous Tskm once
        if Tskm_prev is None:
            Tskm_prev = float(Tskm)
        Thy = float(T[0])  # hypothalamus / core controller node: MATLAB T(1) -> Python T[0]
    
        # --- time step in seconds ---
        dt_sec = float(dTime)        # if dTime is seconds
        # dt_sec = 60.0 * float(dTime)  # uncomment if dTime is minutes
    
    
        # deviations (negative when colder than setpoint)
        dTskm = float(Tskm) - float(Tskm_set)   # ΔT_sk,m
        dThy  = Thy - float(Thy_set)            # ΔT_hy
    
        # skin temp rate (°C/min), then take cooling-only part (negative)
        
        dTskm_dt_min  = 60.0 * ((float(Tskm) - float(Tskm_prev)) / dt_sec)
        dTskm_dt_cool = min(0.0, dTskm_dt_min)  # (dT/dt)^(-)
    
    
        # ---- Fiala signals from your figure ----
        SH = (10.0 * (np.tanh(0.48*dTskm + 3.62) - 1.0) * dTskm -27.9 * dThy - 1.5*109.8* dTskm_dt_cool - 28.6)
        #SH = (30* dThy*dTskm - 3*109.8* dTskm_dt_cool)

        #SH = 109.8* dTskm_dt_cool + 24.36 * (-ERR[0]) * CLDS
    
        VC = (35.0 * (np.tanh(0.34*dTskm + 1.07) - 1.0) * dTskm + 3.9 * dTskm * dTskm_dt_cool)
        #VC = max((-10.8 * ERR[0]) + (-10.8 * (WRMS - CLDS)), 0.0) #JOS3
        #VC = (max(abs((-5.0 * ERR[0]) + ((WRMS - CLDS))), 0.0)) #Stolwijk
        VD = (21.0 * (np.tanh(0.79*dTskm - 0.70) + 1.0) * dTskm + 32.0 * (np.tanh(3.29*dThy - 1.46) + 1.0) * dThy)
        #VD = max((100.5 * ERR[0]) + (6.4 * (WRMS - CLDS)), 0.0) #JOS3
        #VD = max((117.0 * ERR[0]) + (7.5 * (WRMS - CLDS)), 0.0) #Stolwijk
        SW = max((371.2 * ERR[0]) + (33.64 * (WRMS - CLDS)), 0.0) 
        #SW = ((0.8 * np.tanh(0.59*dTskm - 0.19) + 1.2) * dTskm  + (5.7 * np.tanh(1.98*dThy - 1.03) + 6.3) * dThy)
        
        #VD = max((117.0 * ERR(1)) + (7.5 * (WRMS - CLDS)),0);%VD signal Stolwijk -1971 [-]  #1971: CDIL*ERROR(l)  + SDIL* (WARMS-COLDS)  +PDIL*WARM(l*)W ARMS
        #VC =max(-5.0 * (ERR(1) + WRMS - CLDS),0);%VC signal Stolwijk -1971 [-] #-CCON * ERROR (1) -SCON* (WARMS - COLDS) + PCON *COLD (1) * COLDS

        # ---- enforce sign conventions (typical) ----
        # Shivering only when skin below setpoint:
        if dTskm >= 0.0:
            SH = 0.0
    
        # Clamp / rectify as needed
        Shivering = float(np.clip(SH, 0.0, 350.0))
        VD = max(0.0, float(VD))
        VC = max(0.0, float(VC))
        SW = max(0.0, float(SW))
        
        #fx = Shivering#Tskm-273.15#- 109.8* dTskm_dt_cool#- 109.8* dTskm_dt_cool
        # update memory
        Tskm_prev = float(Tskm)
    
    else:
        raise ValueError("active_model must be 'stolwijk' or 'fiala'")

    
    #Original was 
    #VD = max((117.0 * ERR(1)) + (7.5 * (WRMS - CLDS)),0);%VD signal Stolwijk -1971 [-]  #1971: CDIL*ERROR(l)  + SDIL* (WARMS-COLDS)  +PDIL*WARM(l*)W ARMS
    #VC =max(-5.0 * (ERR(1) + WRMS - CLDS),0);%VC signal Stolwijk -1971 [-] #-CCON * ERROR (1) -SCON* (WARMS - COLDS) + PCON *COLD (1) * COLDS

    # ---- Skin blood flow (per segment) ----
    BF_skin = np.zeros(6, dtype=float)

    # Fiala distribution factors (define once; don't redefine inside loop)
    DF_VD_fiala = np.array([0.1320, 0.3020, 0.1150, 0.1210, 0.2300, 0.1000], dtype=float)
    DF_VC_fiala = np.array([0.0500, 0.0207, 0.2367, 0.1150, 0.3785, 0.1850], dtype=float)

    for i in range(6):
        skin_idx = 4 * i + 3  # skin node index

        if skinBF_model.lower() == "stolwijk":
            BF_skin[i] = ((1.0 + (DF_VD[i] * VD)) / (1.0 + DF_VC[i] * VC)) * BFB[skin_idx] * (2.0 ** (ERR[skin_idx] / 6.0))
        elif skinBF_model.lower() == "fiala":
            BF_skin[i] = ((BFB[skin_idx] + DF_VD_fiala[i] * VD) / (1.0 + DF_VC_fiala[i] * VC * np.exp(-VD / 80.0)))                          * (2.0 ** (ERR[skin_idx] / 10.0))
        else:
            raise ValueError("skinBF_model must be 'stolwijk' or 'fiala'")

        #BF_skin[i] = min(BF_skin[i], 90.0)  # cap [L/h]

    #----------------------Shivering-------------------------------------------
    # =============================================================================
    # SHIVERING CONTROLLER
    # =============================================================================

    Q_shiv = DF_SH * Shivering           #Heat generation caused by shivering [W]

    #----------------------Sweating--------------------------------------------
    Psat_sk = np.zeros(6)
    for i in range(6):
        skin_idx = 4 * i + 3 #To identify the skin nodes [-]
        TskinK = T[skin_idx]
        Psat_sk[i] = np.exp(77.3450 + 0.0057 * TskinK - 7235.0 / TskinK) / (TskinK ** 8.2) #Saturation pressure of water vapour at skin [Pa]

    h_evap = 1.0 / (Re_cl + 1.0 / (f_cl * h_conv * L_c))            # [W m^-2 Pa^-1] Evaporative mass transfer coefficient 
    Q_sweat_max = h_evap * (Psat_sk - P_air) * SA                    # [W] Maximum possible evaporative cooling at skin given environmental and clothing properties
    
    #SW = max((320.0 * ERR(1)) + (29.0 * (WRMS - CLDS)),0);%SW signal [W] Stolwijk -1971

    Lh_vap_sweat = 2.426e6      #[J·kg]	Enthalpy / heat of vaporization of Sweat heat at 30⁰C.
    m_SW2 = (SW/Lh_vap_sweat)*3600 # sweat secretion rate per segment [l/hr]
    
    Q_sweat_est = np.zeros(6)
    Q_sweat_capped = np.zeros(6)
    Q_sweat = np.zeros(6)
    Skin_wettedness = np.zeros(6)
    m_SW = np.zeros(6)
    m_EV = np.zeros(6)
    
    
    for i in range(6):
        skin_idx = 4 * i + 3
        #First Q_sweat: Cooling purely physiological possibility from sweat rate of the model
        Q_sweat_est = (DF_SW[i] * SW ) * (2.0 ** (ERR[skin_idx] / 10.0)) #[W] Evaporative heat loss of the skin at each segment as sweating signal response (pag 31 Stolwijk)
        
        #Second Q_sweat: checking that this evaporative cooling is realistic and not exceeding Emax
        #(First 2 Qsweat are to estimate Skin wettedness?)
        Q_sweat_capped[i] = min(Q_sweat_est, Q_sweat_max[i]) # [W] Maximum evaporative cooling at skin provided by SW [W] Page 14 in stolwikj, q_try can't be higher than the max Emax
        
        m_SW[i] = (Q_sweat_capped[i]/Lh_vap_sweat)*3600 #sweat secretion rate per segment [l/hr]
        Skin_wettedness[i] = 0.06 + (0.94 * (Q_sweat_capped[i] / Q_sweat_max[i])) if Q_sweat_max[i] > 0 else 0.06 #Skin wettedness [-] Page 14 Stolwijk but accounting that 0.06 of skin wettedness is not related with sweating (it is a baseline).
        
        #Evaporative cooling at skin provided by SW [W]
        
        #Third Q_sweat (overwritten in matlab code): real evaporative cooling depending on pressure gradient and wind speed (h_evap)
        Q_sweat[i] = Skin_wettedness[i] * Q_sweat_max[i] # Final evaporative cooling using the estimate skin wettedness
        m_EV[i] = (Q_sweat[i]/Lh_vap_sweat)*3600 #sweat evaporation rate per segment c[l/hr]


    m_SW_total = np.sum(m_SW) #Total sweat secretion [l/h]
    m_EV_total = np.sum(m_EV) #Total sweat evaporation [l/h]

    Total_sweat_secretion = np.max([m_SW2, m_SW_total, m_EV_total])# [l/h]
    # print('Remover variable que sea redundante... m_EV_total')
    Total_sweat_evaporation = m_EV_total #[g/s]

    # -------------------------------------------------
    # ----------Heat generation------------------------
    # (Theory from stolwjik 1971: this part of the code has elements of Assign effector output AND calculate heat flows)
    # -------------------------------------------------
    # Effective heat generation by physical activity
    # Q_work is the total heat production as one would determine from the oxygen uptake minus the basal metabolic rate and the caloric equivalent of the external work performed (Pag 27, stolwijk)
   
    Q_total = met * 58.15 * SA_total  #converting the metabolic rate into [W] from [met]
    
    if Q_total - 86.45 > 0.0:         #The model says first 86.45 W is just maintenance metabolism (resting heat generation).
        Q_work = (Q_total - 86.45) * 0.78 #[W]  Effective heat generated by muscles
        # The 0.78 should come from muscle efficiency and energy partitioning. 78% of the energy above basal becomes heat (inefficient muscle contraction)
    else:
        Q_work = 0.0
    Q_work_seg = Q_work * DF_work #[W] muscular work done?

    # Effective heat generation and blood flow at each node (from 1971: Assign effector output?)
    # -------------------------------------------------

    #(Initialization arrays for differential equation solving)
    Q = np.zeros(25)
    BF = np.zeros(24)  # only 24 tissue nodes; central blood (indice 24) handled separately
    Q_TC = np.zeros(25)

    BF_seg = np.zeros(6)
    
    for i in range(6):
        base = 4 * i
    
        # ---------------------------
        # Metabolic heat generation
        # ---------------------------
    
        # core
        Q[base + 0] = Q_bm[base + 0]
        BF[base + 0] = BFB[base + 0]
    
        # muscle
        Q[base + 1] = Q_bm[base + 1] + Q_shiv[i] + Q_work_seg[i]
        BF[base + 1] = BFB[base + 1] + (Q[base + 1] - Q_bm[base + 1]) / 1.073 #1.163
    
        # fat
        Q[base + 2] = Q_bm[base + 2]
        BF[base + 2] = BFB[base + 2]
    
        # skin
        Q[base + 3] = Q_bm[base + 3]
        BF[base + 3] = BF_skin[i] 
    
        # ---------------------------
        # Segment total blood flow
        # ---------------------------
        BF_seg[i] = (
            BF[base + 0]
            + BF[base + 1]
            + BF[base + 2]
            + BF[base + 3]
        )
    
        # ---------------------------
        # Conduction: core → muscle → fat → skin
        # ---------------------------
        Q_TC[base + 0] = TC[base + 0] * (T[base + 0] - T[base + 1])
        Q_TC[base + 1] = TC[base + 1] * (T[base + 1] - T[base + 2])
        Q_TC[base + 2] = TC[base + 2] * (T[base + 2] - T[base + 3])
        Q_TC[base + 3] = 0.0                                                 # [W] Conductive heat transfer from environment to skin 

    # Effective  heat transfer between central blood node and each segment (Through artery/vein)
    # -------------------------------------------------
    Q_bf = np.zeros(24)
    T_cb = T[24] #Temperature of central blood node
    
    if ar_ve_network.lower() == "no":
        for j in range(24):
            Q_bf[j] = 1.163 * BF[j] * (T[j] - T_cb) #Heat transfer by blood flow [W]
            
        Q[24] = Q_bf.sum()  #Net heat at central blood pool [W] - JOS3 #


    elif ar_ve_network.lower() == "yes":
        # ===================== PART 1: perfusion heat from tissues (your first loop) =====================
        # Inputs assumed present:
        # BF (len>=24), T (len>=25 if you use T[24] as pool), T_art (len=6)
        # Allocate:
        Q_bf = np.zeros(24, dtype=float)
        Q_tissue_bf = np.zeros(6, dtype=float)
        
        rhoCp = 1.0501
        
        for j in range(6):                 # j = 0..5  (MATLAB 1..6)
            k = 4 * j                      # MATLAB k = 4*j-3 -> Python 0,4,8,12,16,20
        
            Q_bf[k]   = rhoCp * BF[k]   * (T[k]   - T_art[j])
            Q_bf[k+1] = rhoCp * BF[k+1] * (T[k+1] - T_art[j])
            Q_bf[k+2] = rhoCp * BF[k+2] * (T[k+2] - T_art[j])
            Q_bf[k+3] = rhoCp * BF[k+3] * (T[k+3] - T_art[j])
        
            Q_tissue_bf[j] = Q_bf[k] + Q_bf[k+1] + Q_bf[k+2] + Q_bf[k+3]
        
        
        # ===================== PART 2: FIALA-STYLE CCX artery/vein/pool (your big block) =====================
        # Inputs assumed present:
        # BF (len>=24), BF_seg (len=6), T (len>=25; pool is T[24]), T_art (len=6),
        # hx_ccx (len=6), C (len>=25; pool capacitance is C[24]), dTime (seconds),
        # epsBF small number like 1e-12
        #
        # Outputs written:
        # Q_art (len=6), Qx_ccx (len=6), Q_vein (len=6 with 1,2,3,5 filled), Q[24]
        #
        # NOTE: This translation preserves your MATLAB logic, including the "0.9" factors.
        
        epsBF = 1e-12  # or use your own
        
        rhoCp = 1.0501
        Tpool = T[24]  # MATLAB T(25)
        
        # ---- (0) Arterial flow bookkeeping on your trunk network ----
        BF_art = np.zeros(6, dtype=float)
        BF_art[0] = BF_seg[0]
        BF_art[1] = BF_seg[1]
        BF_art[2] = BF_seg[2] + BF_seg[3]   # pool -> node3 supplies seg3 and branch to seg4
        BF_art[3] = BF_seg[3]              # node3 -> node4
        BF_art[4] = BF_seg[4] + BF_seg[5]  # pool -> node5 supplies seg5 and branch to seg6
        BF_art[5] = BF_seg[5]              # node5 -> node6
        
        # ---- (A) Eq (6): venous PRE-CCX temperature from perfusion-weighted tissue temps ----
        T_vein_pre = np.zeros(6, dtype=float)
        for j in range(6):
            k = 4 * j
            BFsum_layers = max(BF[k] + BF[k+1] + BF[k+2] + BF[k+3], epsBF)
            T_vein_pre[j] = (BF[k] * T[k] + BF[k+1] * T[k+1] + BF[k+2] * T[k+2] + BF[k+3] * T[k+3]) / BFsum_layers
        
        # ---- (B) Arterial advection (network transport) ----
        Q_art = np.zeros(6, dtype=float)
        Q_art[0] = rhoCp * BF_art[0] * (Tpool    - T_art[0])  # pool -> 1
        Q_art[1] = rhoCp * BF_art[1] * (Tpool    - T_art[1])  # pool -> 2
        Q_art[2] = rhoCp * BF_art[2] * (Tpool    - T_art[2])  # pool -> 3
        Q_art[3] = rhoCp * BF_art[3] * (T_art[2] - T_art[3])  # 3 -> 4
        Q_art[4] = rhoCp * BF_art[4] * (Tpool    - T_art[4])  # pool -> 5
        Q_art[5] = rhoCp * BF_art[5] * (T_art[4] - T_art[5])  # 5 -> 6
        
        # ---- (C) Eq (7): CCX heat exchange (paper form) ----
        Qx_ccx = np.zeros(6, dtype=float)
        for j in range(6):
            hx = max(hx_ccx[j], 0.0)
            Qx_ccx[j] = hx * (T_art[j] - T_vein_pre[j])
        
        # ---- (D) Update arterial temperatures with advection +/- branch loss AND CCX sink ----
        # MATLAB:
        # C_art = [(BF_art(1)/sum(BF_art))*C(25), ...]
        BF_art_sum = max(np.sum(BF_art), epsBF)
        C_art = (BF_art / BF_art_sum) * C[24]
        
        T_art_new = T_art.copy()
        
        T_art_new[0] = T_art[0] + ((+Q_art[0]                 - Qx_ccx[0]) / C_art[0]) * dTime
        T_art_new[1] = T_art[1] + ((+Q_art[1]                 - Qx_ccx[1]) / C_art[1]) * dTime
        T_art_new[2] = T_art[2] + ((+Q_art[2] - Q_art[3]      - Qx_ccx[2]) / C_art[2]) * dTime
        T_art_new[3] = T_art[3] + ((+Q_art[3]                 - Qx_ccx[3]) / C_art[3]) * dTime
        T_art_new[4] = T_art[4] + ((+Q_art[4] - Q_art[5]      - Qx_ccx[4]) / C_art[4]) * dTime
        T_art_new[5] = T_art[5] + ((+Q_art[5]                 - Qx_ccx[5]) / C_art[5]) * dTime
        
        T_art[:] = T_art_new  # commit updated arterial temps for THIS step
        
        # ---- (E) Eq (5): venous POST-CCX temperature (per-element form) ----
        Tup = np.zeros(6, dtype=float)
        Tup[[0, 1, 2, 4]] = Tpool          # indices [1 2 3 5] in MATLAB -> [0 1 2 4] in Python
        Tup[3] = T_art[2]                  # upstream for 4 is parent artery 3 (after update)
        Tup[5] = T_art[4]                  # upstream for 6 is parent artery 5 (after update)
        
        T_vein_post = np.zeros(6, dtype=float)
        for j in range(6):
            T_vein_post[j] = T_vein_pre[j] + (Tup[j] - T_art[j])
        
        # ---- (F) Venous mixing (enthalpy-conserving) ----
        BF_3tot = BF_seg[2] + BF_seg[3]
        BF_5tot = BF_seg[4] + BF_seg[5]
        BF_3tot_safe = max(BF_3tot, epsBF)
        BF_5tot_safe = max(BF_5tot, epsBF)
        
        T_vein_3mix = (BF_seg[2] * T_vein_post[2] + BF_seg[3] * T_vein_post[3]) / BF_3tot_safe
        T_vein_5mix = (BF_seg[4] * T_vein_post[4] + BF_seg[5] * T_vein_post[5]) / BF_5tot_safe
        
        # ---- (G) Venous return heat flows into pool ----
        Q_venret_1 = rhoCp * BF_seg[0] * (T_vein_post[0] - Tpool)
        Q_venret_2 = rhoCp * BF_seg[1] * (T_vein_post[1] - Tpool)
        Q_venret_3 = rhoCp * BF_3tot * (T_vein_3mix - Tpool)
        Q_venret_5 = rhoCp * BF_5tot * (T_vein_5mix - Tpool)
        
        Q_vein = np.zeros(6, dtype=float)
        Q_vein[0] = Q_venret_1
        Q_vein[1] = Q_venret_2
        Q_vein[2] = Q_venret_3
        Q_vein[4] = Q_venret_5
        
        # ---- pool RHS ----
        # MATLAB:
        # Q(25) = -(Q_art(1)+Q_art(2)+Q_art(3)+Q_art(5)) + (Q_venret_1+Q_venret_2+Q_venret_3+Q_venret_5)
        Q[24] = -(Q_art[0] + Q_art[1] + Q_art[2] + Q_art[4]) + (Q_venret_1 + Q_venret_2 + Q_venret_3 + Q_venret_5)
        

    # Net heat at tissues
    # -------------------------------------------------

    for i in range(6):
        base = 4 * i
        seg = i
        # sensible & latent heat loss from skin used later
        # Net balances:
        Q[base + 0] = Q[base + 0] - Q_bf[base + 0] - Q_TC[base + 0] # [W] Net heat gain/loss at core 
        Q[base + 1] = Q[base + 1] - Q_bf[base + 1] + Q_TC[base + 0] - Q_TC[base + 1] # [W] Net heat gain/loss at muscle 
        Q[base + 2] = Q[base + 2] - Q_bf[base + 2] + Q_TC[base + 1] - Q_TC[base + 2] # [W] Net heat gain/loss at fat
        Q[base + 3] = Q[base + 3] - Q_bf[base + 3] + Q_TC[base + 2] - Q_sweat[seg] - (h_total[seg] * (T[base + 3] - T_o[seg]) * SA[seg]) # [W] Net heat gain/loss at skin 

    # Sensible and latent heat loss from skin (for post-processing)
    SHL_sk = np.zeros(6)
    LHL_sk = np.zeros(6)
    THL_sk = np.zeros(6)
    for i in range(6):
        skin_idx = 4 * i + 3
        SHL_sk[i] = (h_total[i] * (T[skin_idx] - T_o[i]) * SA[i])
        LHL_sk[i] = Q_sweat[i]
        THL_sk[i] = SHL_sk[i] + LHL_sk[i]

    # Respiratory heat loss
    res_sh = 0.0014 * ((34.0 + 273.15) - T_air_K)            # [W] Respiratory sensible heat loss
    res_lh = 0.0173 * (5.87 - (P_air / 1000.0))              # [W] Respiratory latent heat loss
    RES = (res_sh + res_lh) * Q_total                        # [W] Total respiratory heat loss [W]
    # Apply to trunk core (index 4) and central blood
    Q[4] = Q[4] - RES                                        # [W] Net heat at core node of trunk

    # --------------------------------------------------------------------------
    # Updated temperature based on net heat balance and Capacitance of each node
    # Differential equation solver: Temperature update (Euler explicit fordward integration, like MATLAB code)
    T_new = T + (Q / C) * dTime # [degC] Updated temperature based on net heat balance and Capacitance 
    # --------------------------------------------------------------------------
    
# =============================================================================
#     # --------------------------------------------------------------------------
#     # Euler update ONLY tissue nodes
#     # --------------------------------------------------------------------------
#     T_new = T.copy()
#     T_new[:24] = T[:24] + (Q[:24] / C[:24]) * dTime
#     
#     # --------------------------------------------------------------------------
#     # Fiala-style equilibrium blood pool (dimensionally correct)
#     # betaV := rhoCp * BF_layer  (W/K) consistent with your Q_bf formulation
#     # alpha := S/(hx+S)
#     # Tblp := sum(alpha * sum(betaV*T)) / sum(alpha * sum(betaV))
#     # --------------------------------------------------------------------------
#     rhoCp = 1.0501
#     epsBF = 1e-12
#     
#     num = 0.0
#     den = 0.0
#     
#     for i in range(6):
#         k = 4 * i
#     
#         # S_i = sum(betaV) [W/K]
#         BFsum = BF[k] + BF[k+1] + BF[k+2] + BF[k+3]
#         S = rhoCp * max(BFsum, epsBF)
#     
#         hx = hx_ccx[i]
#         if hx < 0.0:
#             hx = 0.0
#     
#         alpha = S / max(hx + S, epsBF)
#     
#         # sum(betaV*T) [W]
#         sum_betaVT = rhoCp * (
#             BF[k]   * T_new[k]   +
#             BF[k+1] * T_new[k+1] +
#             BF[k+2] * T_new[k+2] +
#             BF[k+3] * T_new[k+3]
#         )
#     
#         num += alpha * sum_betaVT
#         den += alpha * S          # <-- THIS is the missing W/K normalization
#     
#     T_new[24]= num / max(den, epsBF)
#     fx=num / max(den, epsBF) 
#     #print("Tblp =", T_new[24], "K  (", T_new[24]-273.15, "°C )")
# 
# =============================================================================

    # ============================================================================== 

    
    # --------------------------------------------------------------------------
    # Post-processing to estimate: 
    # Mean skin temp, wettedness, total evaporative cooling
    # --------------------------------------------------------------------------

    #Arrays to initialize a area weigthed average
    T_ms_parts = np.zeros(6)    
    To_mean_parts = np.zeros(6) 
    w_sk_parts = np.zeros(6)    
    
    for i in range(6):
        skin_idx = 4 * i + 3
        T_ms_parts[i] = T_new[skin_idx] * (SA[i] / SA_total)
        To_mean_parts[i] = T_o[i] * (SA[i] / SA_total)
        w_sk_parts[i] = Skin_wettedness[i] * (SA[i] / SA_total)
        
    T_ms = T_ms_parts.sum()       # [K] mean skin temperature
    To_mean = To_mean_parts.sum() # [K] mean operating temperature
    w_sk_mean = w_sk_parts.sum()  # [-] mean skin wettedness

    Qsk_latent_total = LHL_sk.sum()     # [W] Total latent heat loss from skin
    Qsk_sensible_total = SHL_sk.sum()   # [W] Total sensible heat loss from skin
    CO = BF.sum()                       # [l/h] Cardiac output 
    Q_storage = (Q_total - (0.22 * Q_work) - LHL_sk.sum() - SHL_sk.sum() - RES) #Heat storage [W]

    
    # --------------------------------------------------------------------------
    # Data output settings
    # --------------------------------------------------------------------------

    # Data output at interval
    if (stepIndex % outputEverySteps) == 0:
        idx = resultIndex

        TimeStamp[idx] = currentTime #[s]

        # Physiological parameters
        T_output[idx, :] = T_new - 273.15       # Temperature at each node [degC]
        w_sk_output[idx, 0] = w_sk_mean         # Mean skin wettedness [-]
        Q_storage_output[idx, 0] = Q_storage    # Heat storage [W]
        T_ms_output[idx, 0] = T_ms - 273.15     # Mean skin temperature[degC]
                                                
        BF_skin_output[idx,0] =  BF_skin.sum()/SA.sum()  # Skin blood flow [l/hr.m2]
        CO_output[idx, 0] = CO                  # Cardiac output [l/hr]
        
        Skin_wettedness_output[idx, :] = Skin_wettedness  # Skin wettedness of individual segment [-]
        BF_output[idx, :] = BF                  # blod flow to each node [l/hr]
        Total_sweat_secretion_output[idx,0] = m_SW_total #Total sweat secretion [l/hr]
        Total_sweat_evaporation_output[idx,0] = m_EV_total #Total sweat evaporation [l/hr]


        # Heat exchange with environment
        Qsk_latent_total_output[idx, 0] = Qsk_latent_total       #%Total latent heat loss[W]
        Qsk_sensible_total_output[idx, 0] = Qsk_sensible_total   #Total sensible heat loss[W]
        SHL_sk_output[idx, :] = SHL_sk                           #Sensible heat loss for individual body segment [W]
        LHL_sk_output[idx, :] = LHL_sk                           # Latent heat loss for individual body segment [W]
        THL_sk_output[idx, :] = THL_sk                           # Total heat loss for individual body segment [W]
        RES_output[idx, 0] = RES                                 # %Respiratory heat loss[W]

        # Environmental / activity
        Tair_output[idx, 0] = T_air                  #[degC]
        MRT_output[idx, 0] = MRT                     #[degC]
        RH_air_output[idx, 0] = RH_air               #[-] fraction
        To_output[idx, :] = T_o - 273.15              #Operative temperature for individual body segment [degC]
        To_mean_output[idx, 0] = To_mean - 273.15    #[degC]
        v_air_output[idx, 0] = v_air                 #[m/s]
        met_output[idx, 0] = met                     #[met]       
        Shivering_output[idx, 0] = Shivering
        VD_output[idx, 0] = VD
        VC_output[idx, 0] = VC
        SW_output[idx, 0] = SW
        #fx_output[idx, 0] = fx
        
        # Increment the result counter
        resultIndex += 1

    # Update for next time step
    T = T_new #[degC]



# Trim arrays to actual number of rows
n = resultIndex
TimeStamp = (TimeStamp[:n] / 60.0)  # minutes
# -------------------------------------------------
# ---------Compute NetQ_storage_output-------------
# -------------------------------------------------
NetQ_storage_output = np.zeros(n)
for i in range(n):
    NetQ_storage_output[i] = Q_storage_output[:i+1, 0].sum()

# -------------------------------------------------
# ---------Write output to Excel-------------------
# -------------------------------------------------
# Build DataFrames (pandas needs 2D columns; expand vectors)
def df_with_time(colname, arr):
    return pd.DataFrame({"TimeStamp": TimeStamp, colname: arr[:n, 0]})

def df_wide(prefix, arr, count):
    cols = {f"{prefix}{i+1}": arr[:n, i] for i in range(count)}
    cols = {"TimeStamp": TimeStamp, **cols}
    return pd.DataFrame(cols)

sheet1 = pd.DataFrame({
    "TimeStamp": TimeStamp,
    "Mean skin temperature (°C)": T_ms_output[:n, 0],
    "Core temperature (°C)": T_output[:n, 4],  # Trunk core is node #5 in MATLAB (index 4)
    "Skin wettedness (mean) (-)": w_sk_output[:n, 0],
    "CO_output (l/h)": CO_output[:n, 0]
})

sheet2 = pd.DataFrame({
    "TimeStamp": TimeStamp,
    "Air temperature (°C)": Tair_output[:n, 0],
    "Mean radiant temperature (°C)": MRT_output[:n, 0],
    "Operating temperature (mean) (°C)": To_mean_output[:n, 0],
    "Relative humidity (-)": RH_air_output[:n, 0],
    "Air speed (m/s)": v_air_output[:n, 0],
    "Metabolic rate (met)": met_output[:n, 0]
})

sheet3 = pd.DataFrame({
    "TimeStamp": TimeStamp,
    "Sensible heat loss from skin (W)": Qsk_sensible_total_output[:n, 0],
    "Latent heat loss from skin (W)": Qsk_latent_total_output[:n, 0],
    "Respiratory heat loss (W)": RES_output[:n, 0],
    "Total Heat storage (W)": Q_storage_output[:n, 0]
})

# Local temperature (25 nodes)
temp_cols = {f"T{i+1} (°C)": T_output[:n, i] for i in range(25)}
sheet4 = pd.DataFrame({"TimeStamp": TimeStamp, **temp_cols})

# Skin wettedness per segment (6)
sheet5 = df_wide("Skin_wettedness seg ", Skin_wettedness_output, 6)

# Sensible heat loss (6)
sheet6 = df_wide("SHL skin seg ", SHL_sk_output, 6)

# Latent heat loss (6)
sheet7 = df_wide("LHL skin seg ", LHL_sk_output, 6)

# Blood flow (24)
sheet8 = df_wide("BF node ", BF_output, 24)

with pd.ExcelWriter(BASE_DIR / f"{name_simulation}_output_python.xlsx", engine="openpyxl") as writer:
    sheet1.to_excel(writer, index=False, sheet_name="Summary")
    sheet2.to_excel(writer, index=False, sheet_name="Ambient TCition")
    sheet3.to_excel(writer, index=False, sheet_name="Heat exchange")
    sheet4.to_excel(writer, index=False, sheet_name="Local temperature")
    sheet5.to_excel(writer, index=False, sheet_name="Local skin wettedness")
    sheet6.to_excel(writer, index=False, sheet_name="Local sensible heat loss")
    sheet7.to_excel(writer, index=False, sheet_name="Local latent heat loss")
    sheet8.to_excel(writer, index=False, sheet_name="Local blood flow")



# -------------------------------------------------
# -------------------Plots--------------------------
# -------------------------------------------------

# # =========================
# # Environmental parameters
# # =========================

# fig = plt.figure(figsize=(8,6))
# gs = gridspec.GridSpec(3, 2, hspace=0.5)

# # 1) Air temperature
# ax = fig.add_subplot(gs[0])
# ax.plot(TimeStamp, Tair_output, linewidth=1.5)
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Air Temperature (°C)")
# ax.set_title("Air Temperature", fontsize=8)
# ax.grid(True, alpha=0.3)

# # 2) Mean radiant temperature
# ax = fig.add_subplot(gs[1])
# ax.plot(TimeStamp, MRT_output, linewidth=1.5)
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Mean radiant temperature (°C)")
# ax.set_title("Mean radiant temperature", fontsize=8)
# ax.grid(True, alpha=0.3)

# # 3) Relative humidity
# ax = fig.add_subplot(gs[2])
# ax.plot(TimeStamp, RH_air_output, linewidth=1.5)
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Relative humidity (-)")
# ax.set_title("Relative humidity", fontsize=8)
# ax.grid(True, alpha=0.3)

# # 4) Air speed
# ax = fig.add_subplot(gs[3])
# ax.plot(TimeStamp, v_air_output, linewidth=1.5)
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Air speed (m/s)")
# ax.set_title("Air speed", fontsize=12)
# ax.grid(True, alpha=0.3)

# # 5) Mean operating temperature
# ax = fig.add_subplot(gs[4])
# ax.plot(TimeStamp, To_mean_output, linewidth=1.5)
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Operating temperature (°C)")
# ax.set_title("Operating temperature", fontsize=8)
# ax.grid(True, alpha=0.3)

# # 6) Activity levels during the run
# ax = fig.add_subplot(gs[5])
# ax.plot(TimeStamp, met_output, linewidth=1.5)
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Metabolic heat (METs)")
# ax.set_title("Activity level", fontsize=8)
# ax.grid(True, alpha=0.3)

# fig.suptitle("Environmental parameters", fontsize=12,y=0.96)  
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# #plt.savefig(workdir + name_simulation + '_Input_data.png',dpi=400,transparent=False,bbox_inches='tight')   
# plt.show()


# fig = plt.figure(figsize=(8,4))
# gs = gridspec.GridSpec(2, 2, hspace=0.4)

# # 1) Skin and core temperature
# ax = fig.add_subplot(gs[0])
# ax.plot(TimeStamp, T_output[:, 4], linewidth=1.5, label="Core Temperature (Chest)", color = 'Maroon')
# ax.plot(TimeStamp, T_ms_output, linewidth=1.5, label="Mean Skin Temperature", color = 'Black')
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Temperature (°C)")
# plt.legend(loc=3)

# # 4) Skin blood flow
# ax = fig.add_subplot(gs[1])
# ax.plot(TimeStamp, BF_skin_output, linewidth=1.5, label="All body: Skin blood flow ", color = 'Black')
# ax.set_xlabel("Time (min)")
# ax.set_ylabel(" Skin blood flow (l/h.m2)")
# # 6-8 l/min https://www.mayoclinicproceedings.org/article/S0025-6196(11)61930-7/fulltext
# #Tartarini 2022... 80 L/(hm2)

# ax2 = ax.twinx()
# ax2.plot(TimeStamp,CO_output, label="Cardiac output", color = 'Green')
# ax2.set_ylabel("Cardiac output (L/h)")

# lines, labels = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# # Create a single legend using the combined lists
# ax.legend(lines + lines2, labels + labels2)


# # 2) Mean skin wettedness
# ax = fig.add_subplot(gs[2])
# ax.plot(TimeStamp, w_sk_output, linewidth=1.5, label="All body: skin wettedness", color = 'Navy')
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Skin wettedness (-)")
# plt.legend()

 
# # 3) Total sweat secreted (and accumulated over time)
# ax = fig.add_subplot(gs[3])
# ax.plot(TimeStamp,Total_sweat_secretion_output , linewidth=1.5, label="Sweat rate secretion", color ='Navy') 
# ax.plot(TimeStamp,Total_sweat_evaporation_output, linewidth=1.5, label="Sweat rate evaporation ", color ='crimson')
# ax.set_ylabel("Sweat rate (l/h)")
# plt.legend()
# ax2 = ax.twinx()
# ax2.plot(TimeStamp,(Total_sweat_secretion_output/60).cumsum(), label="Cumulative sweat secreted", color ='Navy',ls ='--')
# ax2.plot(TimeStamp,(Total_sweat_evaporation_output/60).cumsum(), label="Cumulative sweat evaporated", color ='crimson',ls ='--')
# ax2.set_ylabel("Whole body sweat loss over time (L)")
# ax.set_xlabel("Time (min)")
# plt.legend()

# plt.suptitle("Physiological thermal responses", fontsize=12,y=0.96)
# #plt.savefig(workdir + name_simulation + '_Physiological_responses.png',dpi=400,transparent=False,bbox_inches='tight')   
# plt.show()



# # ===========================================
# # Heat exchange with the environment (totals)
# # ===========================================
# fig  = plt.figure( figsize=(7, 4.5))
# gs = gridspec.GridSpec(3, 2, hspace=0.5)

# # 1) Sensible heat transfer (skin total)
# ax = fig.add_subplot(gs[0])
# ax.plot(TimeStamp, Qsk_sensible_total_output, linewidth=1.5)
# ax.axhline(0, linewidth=1.5, color = '0.5')
# ax.set_xlabel("Time (min)")
# ax.set_ylabel(" Sensible heat transfer (W)")
# ax.set_title("C+R -Sensible heat transfer", fontsize=8)
# ax.grid(True, alpha=0.3)

# # 2) Latent heat transfer (skin total)
# ax = fig.add_subplot(gs[1])
# ax.plot(TimeStamp, Qsk_latent_total_output, linewidth=1.5)
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Latent heat transfer (W)")
# ax.set_title("Emax - Latent heat transfer", fontsize=8)
# ax.grid(True, alpha=0.3)

# # 3) Respiratory heat loss
# ax = fig.add_subplot(gs[2])
# ax.plot(TimeStamp, RES_output, linewidth=1.5)
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Respiratory heat loss (W)")
# ax.set_title("Respiratory heat loss", fontsize=8)
# ax.grid(True, alpha=0.3)

# # 4) Body heat storage (instantaneous)
# ax = fig.add_subplot(gs[3])
# ax.plot(TimeStamp, Q_storage_output, linewidth=1.5)
# ax.axhline(0, linewidth=1.5, color = '0.5')
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("NetQ_storage_output (W)")
# ax.set_title("NetQ_storage_output", fontsize=8)
# ax.grid(True, alpha=0.3)

# fig.suptitle("Heat exchanges", fontsize=12,y=0.96)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# #plt.savefig(workdir + name_simulation + '_Heat_exchanges.png',dpi=400,transparent=False,bbox_inches='tight')   
# plt.show()


# Create standalone figure
plt.figure(figsize=(8,5))

# Plot lines
plt.plot(TimeStamp, T_output[:, 4],
         linewidth=1.5,
         label="Core Temperature (Chest)",
         color='Maroon')

# plt.plot(TimeStamp, T_ms_output,
#          linewidth=1.5,
#          label="Mean Skin Temperature",
#          color='Black')

# Labels
plt.xlabel("Time (min)")
plt.ylabel("Temperature (°C)")

# Legend
plt.legend(loc=3)
plt.savefig(BASE_DIR / f"{name_simulation}_Tcore.png",dpi=1080,transparent=False,bbox_inches='tight')   
# Show plot
plt.show()

# Create standalone figure
plt.figure(figsize=(8,5))

# Plot lines
# plt.plot(TimeStamp, T_output[:, 4],
#          linewidth=1.5,
#          label="Core Temperature (Chest)",
#          color='Maroon')

plt.plot(TimeStamp, fx_output,
          linewidth=1.5,
          label="Mean Skin Temperature",
          color='Black')

# Labels
plt.xlabel("Time (min)")
plt.ylabel("Temperature (°C)")

# Legend
plt.legend(loc=3)
plt.savefig(BASE_DIR / f"{name_simulation}_Tcore.png",dpi=1080,transparent=False,bbox_inches='tight')   
# Show plot
plt.show()
