import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from classes import *
from functions import *

CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2 = [
    Molecule("Methanol", Mw_Me, H_f_Me, Tb_Me, Vb_Me, Param_Mu_Me, Param_Cp_Me, Param_kappa_Me,),
    Molecule("Oxygen",  Mw_O2, H_f_O2,  Tb_O2, Vb_O2, Param_Mu_O2, Param_Cp_O2, Param_kappa_O2),
    Molecule("Formaldehyde", Mw_HCHO, H_f_HCHO, Tb_HCHO, Vb_HCHO, Param_Mu_HCHO, Param_Cp_HCHO, Param_kappa_HCHO,),
    Molecule("Water", Mw_H2O, H_f_H2O, Tb_H2O, Vb_H2O, Param_Mu_H2O, Param_Cp_H2O, Param_kappa_H2O,),
    Molecule("Carbon Monoxide", Mw_CO, H_f_CO, Tb_CO, Vb_CO, Param_Mu_CO, Param_Cp_CO, Param_kappa_CO,),
    Molecule("DME", Mw_DME, H_f_DME, Tb_DME, Vb_DME, Param_Mu_DME, Param_Cp_DME, Param_kappa_DME,),
    Molecule("DMM", Mw_DMM, H_f_DMM, Tb_DMM, Vb_DMM, Param_Mu_DMM, [0, 0, 0, 0], Param_kappa_DMM,),
    Molecule("Nitrogen", Mw_N2, H_f_N2, Tb_N2, Vb_N2, Param_Mu_N2, Param_Cp_N2, Param_kappa_N2,),
]

r1, r2, r3, r4, r5 = [
    Reaction("reaction_1", [1, 0.5], [1, 1], [CH3OH, O2], [HCHO, H2O]),
    Reaction("reaction_2", [1, 0.5], [1, 1], [HCHO, O2], [CO, H2O]),
    Reaction("reaction_3", [2], [1, 1], [CH3OH], [DME, H2O]),
    Reaction("reaction_4", [2, 1], [1, 1], [CH3OH, HCHO], [DMM, H2O]),
    Reaction("reaction_5", [1, 1], [2, 1], [DME, O2], [HCHO, H2O]),
]

m = 400
snaps = 600
t_dur = 300

catalyst_weight = w_cat
dx = catalyst_weight / m

num_vars = 7


def deriv(t, C):
    dCdt = np.zeros_like(C)

    for i in range(1, m):
        
        C_tot = C[i] + C[i + m] + C[i + 2*m] + C[i + 3*m] + C[i + 4*m] + C[i + 5*m] + C[i + 6*m] + C_I0

        dCdt[i] = -u(T_0, P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i] - C[i-1]) / (dx)) - (r1.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + 2*r3.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + 2*r4.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]))

        dCdt[i + m] = -u(T_0, P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + m] - C[i-1 + m]) / (dx)) - 0.5*(r1.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + r2.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + r5.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]))

        dCdt[i + 2*m] = -u(T_0, P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 2*m] - C[i-1 + 2*m]) / (dx)) + (r1.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + r5.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m])) - (r2.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + r4.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]))

        dCdt[i + 3*m] = -u(T_0, P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 3*m] - C[i-1 + 3*m]) / (dx)) + (r1.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + r2.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + r3.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + r4.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) + 0.5*r5.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]))
        
        dCdt[i + 4*m] = -u(T_0, P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 4*m] - C[i-1 + 4*m]) / (dx)) + r2.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m])

        dCdt[i + 5*m] = -u(T_0, P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 5*m] - C[i-1 + 5*m]) / (dx)) + r3.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m]) - 0.5*r5.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m])

        dCdt[i + 6*m] = -u(T_0, P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 6*m] - C[i-1 + 6*m]) / (dx)) + r4.r(T_0, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m])

    # dCdt[0] = 0
    return dCdt


uinit = np.zeros(num_vars*m)
uinit[0] = C_A0
uinit[m] = C_B0
uinit[1:m] = 1e-100

time = np.linspace(0, t_dur, snaps)

sol = solve_ivp(deriv, (0, t_dur), uinit, method='RK45', t_eval=time, atol=1e-12, rtol=1e-12)

x_points = np.linspace(0, catalyst_weight, m)

plt.plot(x_points, sol.y[:m, -1], x_points, sol.y[m:2*m, -1], x_points, sol.y[2*m:3*m, -1], x_points, sol.y[3*m:4*m, -1], x_points, sol.y[4*m:5*m, -1], x_points, sol.y[5*m:6*m, -1], x_points, sol.y[6*m:, -1])

plt.show()

plt.plot(time, sol.y[1])
plt.show()

print(sol.y[3*m - 1, -1])
