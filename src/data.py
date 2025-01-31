R = 8.314  # [J/(mol*K)]
r_inner = 0.05  # [m]
rho_cat = 1000  # [kg/m³]
w_cat = 4.755 # 0.9157 # [kg]
r_part = 3e-3  # [m]
Dpe = r_part / 3 # [m]
T_0 = 170 + 273.15  # [K]
Ts_0 = 240 + 273.15
T_cool = 200 + 273.15 
P_0 = 101325  # [Pa]
u_0 = 1 # [m/s]

UA = 50*(4/(2*r_inner)) # [W / m³ K]

C_A0 = 0.01 * (P_0/(R*T_0))  # [mol/m³]
C_B0 = 0.21 * 0.99 * (P_0/(R*T_0))  # [mol/m³]
C_C0 = 0  # [mol/m³]
C_D0 = 0  # [mol/m³]
C_E0 = 0  # [mol/m³]
C_F0 = 0  # [mol/m³]
C_G0 = 0  # [mol/m³]
C_I0 = 0.79 * 0.99 * (P_0/(R*T_0))  # [mol/m³]
C_T0 = C_A0 + C_B0 + C_I0

q_dot_0 = u_0*3.14*r_inner*r_inner*0.71
print(q_dot_0)

# F_A = C_A0*u_0*3.14*(r_inner**2)
# print(C_A0, F_A)
C_As0 = 1e-10  # [mol/m³]
C_Bs0 = 0  # [mol/m³]  
C_Cs0 = 0  # [mol/m³]  
C_Ds0 = 0  # [mol/m³]  
C_Es0 = 0  # [mol/m³]  
C_Fs0 = 0  # [mol/m³]  
C_Gs0 = 0  # [mol/m³] 

# Data from deshmuk
A_CH3OH = 2.6e-4  # [atm^-1]
A_O2 = 1.423e-5  # [atm^-0.5]
A_H2O = 5.5e-7  # [atm^-1]
A_HCHO = 1.5e7  # [mol/(kg_cat s)]
A_CO = 3.5e2  # [mol/(kg_cat s atm)]
A_DMEf = 1.9e5  # [mol/(kg_cat s atm)]
A_DMMf = 4.26e-6  # [mol/(kg_cat s atm²)]
A_DME = 5e-7  # [atm^-1]
A_DMEHCHO = 6.13e5  # [mol/(kg_cat s)]


Ea_CH3OH = -56780  # [J/mol]
Ea_O2 = -60320  # [J/mol]
Ea_H2O = -86450  # [J/mol]
Ea_HCHO = 86000  # [J/mol]
Ea_CO = 46000  # [J/mol]
Ea_DMEf = 77000  # [J/mol]
Ea_DMMf = 46500  # [J/mol]
Ea_DME = -96720  # [J/mol]
Ea_DMEHCHO = 98730  # [J/mol]

# Perry (except DMM from CRC handbook)
H_f_O2, H_f_N2, H_f_CO, H_f_H2O, H_f_Me, H_f_HCHO, H_f_DME, H_f_DMM = [0, 0, -110.53e3, -241.818e3, -200.94e3, -108.6e3, -184.1e3, -348.5e3]  # J/mol

# PubChem
Mw_Me, Mw_O2, Mw_H2O, Mw_HCHO, Mw_CO, Mw_DME, Mw_DMM, Mw_N2 = [32.042, 32, 18.015, 30.026, 28.010, 46.07, 76.09, 28.014]  # g/mol

# boiling points 1 atm 
Tb_Me, Tb_O2, Tb_H2O, Tb_HCHO, Tb_CO, Tb_DME, Tb_DMM, Tb_N2 = [337.8, 90.188, 373.15, 254, 81.6, 249, 315, 77]  # K

# molar liquid volumes from schroeder properties of liquids and gases 4-10
Vb_Me, Vb_O2, Vb_H2O, Vb_HCHO, Vb_CO, Vb_DME, Vb_DMM, Vb_N2 = [42.7, 21, 18.8, 35, 28, 63, 91, 28]  # cm³/mol

# Perry
# Param_Cp_Me, Param_Cp_O2, Param_Cp_H2O, Param_Cp_HCHO, Param_Cp_CO, Param_Cp_DME, Param_Cp_N2 = [
#     [0.39252e5, 0.87900e5, 1.91650e3, 0.53654e5, 896.7],
#     [0.29103e5, 0.10040e5, 2.52650e3, 0.09356e5, 1153.8],
#     [0.33363e5, 0.26790e5, 2.61050e3, 0.08896e5, 1169],
#     [0.33503e5, 0.49394e5, 1.92800e3, 0.29728e5, 965.04],
#     [0.29108e5, 0.08773e5, 3.08510e3, 0.08455e5, 1538.2],
#     [0.57431e5, 0.94494e5, 0.89551e3, 0.65065e5, 2467.4],
#     [0.29105e5, 0.08615e5, 1.70160e3, 0.00103e5, 909.79],
# ]

# Param_Cp_Me, Param_Cp_O2, Param_Cp_H2O, Param_Cp_HCHO, Param_Cp_CO, Param_Cp_DME, Param_Cp_DMM, Param_Cp_N2 = [
#     [-2.21577299e-05, 9.80583616e-02, 1.62213845e+01],
#     [2.13040626e-06, 7.47568317e-03, 2.67994701e+01],
#     [7.79167516e-06, 2.33956061e-03, 3.21016396e+01],
#     [2.51360384e-06, 4.09555823e-02, 2.26314061e+01],
#     [9.31132309e-06, -3.99925344e-03, 2.94663366e+01],
#     [-5.80524503e-05, 1.87165392e-01, 1.42544699e+01],
#     [-4.47947000e-05, 2.23946668e-01, 4.16167780e+01],
#     [9.09533025e-06, -4.93240794e-03, 2.97781394e+01]
# ]


Param_Cp_Me, Param_Cp_O2, Param_Cp_H2O, Param_Cp_HCHO, Param_Cp_CO, Param_Cp_DME, Param_Cp_DMM, Param_Cp_N2 = [
    [-1.00276518e-07, 1.27956218e-04, 2.55827510e-02, 2.74676988e+01],
    [-3.44533102e-08,  5.37070116e-05, -1.74257069e-02,  3.06635129e+01],
    [-2.18093891e-08,  4.04403306e-05, -1.34233401e-02,  3.45476285e+01],
    [-8.90612882e-08,  1.35838352e-04, -2.34141363e-02,  3.26198986e+01],
    [-1.50405940e-08,  3.18270923e-05, -1.48699563e-02,  3.11531847e+01],
    [-1.79737332e-07,  2.11014335e-04,  5.72588786e-02,  3.44125545e+01],
    [-8.5100e-08,  8.2600e-05,  1.6244e-01,  5.1161e+01],
    [-7.24573709e-09,  1.99421987e-05, -1.01693191e-02,  3.05907707e+01]
]


Param_Mu_Me, Param_Mu_O2, Param_Mu_H2O, Param_Mu_HCHO, Param_Mu_CO, Param_Mu_DME, Param_Mu_N2, Param_Mu_DMM = [
    [3.0663e-7, 0.69655, 205],
    [1.1010e-6, 0.5634, 96.3],
    [1.7096e-8, 1.1146, 0],
    [1.5948e-5, 0.21516, 1151.1],
    [1.1127e-6, 0.5338, 94.7],
    [2.6800e-6, 0.3975, 534],
    [6.5592e-7, 0.6081, 54.714],
    [1.9344, 3.2209e-1, -6.6278e-6, -1.9285e-8]
]

Param_kappa_Me, Param_kappa_O2, Param_kappa_H2O, Param_kappa_HCHO, Param_kappa_CO, Param_kappa_DME, Param_kappa_N2, Param_kappa_DMM = [
    [5.7992e-7, 1.7862, 0, 0],
    [0.00044994, 0.7456, 56.699, 0],
    [6.2041e-6, 1.3973, 0, 0],
    [5.2201e-6, 1.417, 0, 0],
    [0.00059882, 0.6863, 57.13, 501.92],
    [0.059975, 0.2667, 1018.6, 1098800],
    [0.00033143, 0.7722, 16.323, 373.72],
    [-2.7748e-3, 2.9488e-5, 1.1508e-7, -4.5122e-11]
]

dh1 = (H_f_H2O + H_f_HCHO) - (H_f_Me + 0.5*H_f_O2)
dh2 = (H_f_CO + H_f_H2O) - (H_f_HCHO + 0.5*H_f_O2)
dh3 = (H_f_DME + H_f_H2O) - (2*H_f_Me) 
dh4 = (H_f_DMM + H_f_H2O) - (2*H_f_Me + H_f_HCHO)
dh5 = (2*H_f_HCHO + H_f_H2O) - (H_f_DME + H_f_O2)

# print(dh1/1000, dh2/1000, dh3/1000, dh4/1000, dh5/1000)
