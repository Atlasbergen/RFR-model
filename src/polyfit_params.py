import time
from math import exp, sinh, cosh
import matplotlib.pyplot as plt
import numpy as np 
from classes import Molecule
from data import *

MeOH = Molecule("Methanol", Mw_Me, H_f_Me, Tb_Me, Vb_Me, Param_Mu_Me, Param_Cp_Me, Param_kappa_Me)
O2 = Molecule("Oxygen",  Mw_O2, H_f_O2,  Tb_O2, Vb_O2, Param_Mu_O2, Param_Cp_O2, Param_kappa_O2)
HCHO = Molecule("Formaldehyde", Mw_HCHO, H_f_HCHO, Tb_HCHO, Vb_HCHO, Param_Mu_HCHO, Param_Cp_HCHO, Param_kappa_HCHO)
H2O = Molecule("Water", Mw_H2O, H_f_H2O, Tb_H2O, Vb_H2O, Param_Mu_H2O, Param_Cp_H2O, Param_kappa_H2O)
CO = Molecule("Carbon Monoxide", Mw_CO, H_f_CO, Tb_CO, Vb_CO, Param_Mu_CO, Param_Cp_CO, Param_kappa_CO)
DME = Molecule("DME", Mw_DME, H_f_DME, Tb_DME, Vb_DME, Param_Mu_DME, Param_Cp_DME, Param_kappa_DME)
DMM = Molecule("DMM", Mw_DMM, H_f_DMM, Tb_DMM, Vb_DMM, Param_Mu_DMM, [0], Param_kappa_DMM)
N2 = Molecule("Nitrogen", Mw_N2, H_f_N2, Tb_N2, Vb_N2, Param_Mu_N2, Param_Cp_N2, Param_kappa_N2)

x_data = np.linspace(298, 700, 10000)
y_data = [N2.Cp(i) for i in x_data]

coeffs = np.polyfit(x_data, y_data, 3)
y_fit = np.polyval(coeffs, x_data)

plt.plot(x_data, y_data, x_data, y_fit, 'r--')
plt.show()

print(coeffs)

# Mean Squared Error (MSE)
mse = np.mean((y_data - y_fit) ** 2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(y_data - y_fit))

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
