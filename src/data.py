import polars as pl
import numpy as np

R = 8.314  # [J/(mol*K)]
r_inner = 0.0025  # [m]
rho_cat = 700  # [kg/m³]
w_cat = 0.00011  # [kg]
r_part = 150e-6  # [m]
T_0 = 260 + 273.15  # [K]
F_A0 = 1e-4  # [mol/s]
F_B0 = 1e-4  # [mol/S]
F_C0 = 0  # [mol/s]
F_D0 = 0  # [mol/s]
F_E0 = 0  # [mol/s]
F_F0 = 0  # [mol/s]
F_G0 = 0  # [mol/s]
F_I0 = 8e-4  # [mol/s]
F_T0 = F_A0 + F_B0 + F_I0
P_0 = 1  # [atm]

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

df = pl.read_csv("Air_viscosity.csv")  # data found in perry (2-196)

temp_viscosity, air_viscosity = np.array(df[0:, 0]), np.array(df[0:, 1])

temp_cp = np.array([298.15, 400, 500, 600, 800, 1000, 1500])

H_f_O2, H_f_N2, H_f_CO, H_f_H2O, H_f_Me, H_f_HCHO, H_f_DME, H_f_DMM = [0, 0, -110.53e3, -241.818e3, -200.94e3, -108.6e3, -184.1e3, -348.5e3]

H_rxn0_1 = (H_f_HCHO + H_f_H2O) - (H_f_Me + 0.5*H_f_O2)

H_rxn0_2 = (H_f_CO + H_f_H2O) - (H_f_HCHO + H_f_O2)

H_rxn0_3 = (H_f_DME + H_f_H2O) - (2*H_f_Me)

H_rxn0_3_rev = -H_rxn0_3

H_rxn0_4 = (H_f_DMM + H_f_H2O) - (2*H_f_Me + H_f_HCHO)

H_rxn0_4_rev = -H_rxn0_4

H_rxn0_5 = (2*H_f_HCHO + H_f_H2O) - (H_f_DME + H_f_O2)
