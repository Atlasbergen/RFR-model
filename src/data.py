R = 8.314  # [J/(mol*K)]
r_inner = 0.25  # [m]
rho_cat = 1000  # [kg/m³]
w_cat = 1.1  # [kg]
r_part = 150e-6  # [m]
T_0 = 260 + 273.15  # [K]
F_A0 = 1  # [mol/s]
F_B0 = 1  # [mol/S]
F_C0 = 0  # [mol/s]
F_D0 = 0  # [mol/s]
F_E0 = 0  # [mol/s]
F_F0 = 0  # [mol/s]
F_G0 = 0  # [mol/s]
F_I0 = 8  # [mol/s]
F_T0 = F_A0 + F_B0 + F_I0
P_0 = 1  # [atm]

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

# Perry
Param_Cp_Me, Param_Cp_O2, Param_Cp_H2O, Param_Cp_HCHO, Param_Cp_CO, Param_Cp_DME, Param_Cp_N2 = [
    [0.39252e5, 0.87900e5, 1.91650e3, 0.53654e5, 896.7],
    [0.29103e5, 0.10040e5, 2.52650e3, 0.09356e5, 1153.8],
    [0.33363e5, 0.26790e5, 2.61050e3, 0.08896e5, 1169],
    [0.33503e5, 0.49394e5, 1.92800e3, 0.29728e5, 965.04],
    [0.29108e5, 0.08773e5, 3.08510e3, 0.08455e5, 1538.2],
    [0.57431e5, 0.94494e5, 0.89551e3, 0.65065e5, 2467.4],
    [0.29105e5, 0.08615e5, 1.70160e3, 0.00103e5, 909.79],
]

Param_Mu_Me, Param_Mu_O2, Param_Mu_H2O, Param_Mu_HCHO, Param_Mu_CO, Param_Mu_DME, Param_Mu_N2 = [
    [3.0663e-7, 0.69655, 205],
    [1.1010e-6, 0.5634, 96.3],
    [1.7096e-8, 1.1146, 0],
    [1.5948e-5, 0.21516, 1151.1],
    [1.1127e-6, 0.5338, 94.7],
    [2.6800e-6, 0.3975, 534],
    [6.5592e-7, 0.6081, 54.714],
]
