import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as tm
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

m = 10
snaps = 100
t_dur = 2

length = reactor_len(w_cat)
dx = length / m

num_vars = 16 


def deriv(t, C):
    dCdt = np.zeros_like(C)

    for i in range(1, m):
        
        C_tot = C[i] + C[i + m] + C[i + 2*m] + C[i + 3*m] + C[i + 4*m] + C[i + 5*m] + C[i + 6*m] + C_I0

        dCdt[i] = -u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i] - C[i-1]) / (dx)) + k_c(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2 * r_part,
            CH3OH.D_eff(C[i + 7*m], P_0, C_tot, C[i], [C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * A_c(r_inner) * (C[i + 8*m] - C[i])

        dCdt[i + m] = -u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + m] - C[i-1 + m]) / (dx)) + k_c(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2 * r_part,
            O2.D_eff(C[i + 7*m], P_0, C_tot, C[i + m], [C[i], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * A_c(r_inner) * (C[i + 9*m] - C[i + m])

        dCdt[i + 2*m] = -u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 2*m] - C[i-1 + 2*m]) / (dx)) + k_c(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2 * r_part,
            HCHO.D_eff(C[i + 7*m], P_0, C_tot, C[i + 2*m], [C[i + m], C[i], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * A_c(r_inner) * (C[i + 10*m] - C[i + 2*m])

        dCdt[i + 3*m] = -u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 3*m] - C[i-1 + 3*m]) / (dx)) + k_c(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2 * r_part,
            H2O.D_eff(C[i + 7*m], P_0, C_tot, C[i + 3*m], [C[i + m], C[i + 2*m], C[i], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * A_c(r_inner) * (C[i + 11*m] - C[i + 3*m])
        
        dCdt[i + 4*m] = -u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 4*m] - C[i-1 + 4*m]) / (dx)) + k_c(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2 * r_part,
            CO.D_eff(C[i + 7*m], P_0, C_tot, C[i + 4*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * A_c(r_inner) * (C[i + 12*m] - C[i + 4*m])

        dCdt[i + 5*m] = -u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 5*m] - C[i-1 + 5*m]) / (dx)) + k_c(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2 * r_part,
            DME.D_eff(C[i + 7*m], P_0, C_tot, C[i + 5*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * A_c(r_inner) * (C[i + 13*m] - C[i + 5*m])

        dCdt[i + 6*m] = -u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 6*m] - C[i-1 + 6*m]) / (dx)) + k_c(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2 * r_part,
            DMM.D_eff(C[i + 7*m], P_0, C_tot, C[i + 6*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * A_c(r_inner) * (C[i + 14*m] - C[i + 6*m])

        dCdt[i + 7*m] = -u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0) * ((C[i + 7*m] - C[i-1 + 7*m]) / (dx)) + h(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2*r_part,
            Molecule.kappa_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            Molecule.Cp_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            )
        ) * a_c(2*r_inner, 2*r_part) * (1/(CH3OH.Cp(C[i + 7*m])*C[i] + O2.Cp(C[i + 7*m])*C[i + m] + HCHO.Cp(C[i + 7*m])*C[i + 2*m] + H2O.Cp(C[i + 7*m])*C[i + 3*m] + CO.Cp(C[i + 7*m])*C[i + 4*m] + DME.Cp(C[i + 7*m])*C[i + 5*m] + DMM.Cp(C[i + 7*m])*C[i + 6*m] + N2.Cp(C[i + 7*m])*C_I0)) * (C[i + 15*m] - C[i + 7*m])
        
        dCdt[i + 8*m] = k_c(
        rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        Molecule.mu_gas_mix(
            C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        2 * r_part,
        CH3OH.D_eff(C[i + 7*m], P_0, C_tot, C[i], [C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * (C[i] - C[i + 8*m]) - rho_cat*(1-porosity(2*r_inner, 2*r_part))*(r1.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + 2*r3.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + 2*r4.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]))

        dCdt[i + 9*m] = k_c(
        rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        Molecule.mu_gas_mix(
            C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        2 * r_part,
        O2.D_eff(C[i + 7*m], P_0, C_tot, C[i + m], [C[i], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * (C[i + m] - C[i + 9*m]) - 0.5*rho_cat*(r1.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + r2.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + r5.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]))

        dCdt[i + 10*m] = k_c(
        rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        Molecule.mu_gas_mix(
            C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        2 * r_part,
        HCHO.D_eff(C[i + 7*m], P_0, C_tot, C[i + 2*m], [C[i + m], C[i], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * (C[i + 2*m] - C[i + 10*m]) + rho_cat*((r1.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + r5.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])) - (r2.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + r4.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])))

        dCdt[i + 11*m] = k_c(
        rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        Molecule.mu_gas_mix(
            C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        2 * r_part,
        H2O.D_eff(C[i + 7*m], P_0, C_tot, C[i + 3*m], [C[i + m], C[i + 2*m], C[i], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * (C[i + 3*m] - C[i + 11*m]) + rho_cat*(r1.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + r2.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + r3.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + r4.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) + 0.5*r5.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]))

        dCdt[i + 12*m] = k_c(
        rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        Molecule.mu_gas_mix(
            C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        2 * r_part,
        CO.D_eff(C[i + 7*m], P_0, C_tot, C[i + 4*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C[i + 4*m] - C[i + 12*m]) + rho_cat*r2.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])

        dCdt[i + 13*m] = k_c(
        rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        Molecule.mu_gas_mix(
            C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        2 * r_part,
        DME.D_eff(C[i + 7*m], P_0, C_tot, C[i + 5*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * (C[i + 5*m] - C[i + 13*m]) + rho_cat*(r3.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]) - 0.5*r5.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m]))

        dCdt[i + 14*m] = k_c(
        rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        Molecule.mu_gas_mix(
            C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
        2 * r_part,
        DMM.D_eff(C[i + 7*m], P_0, C_tot, C[i + 6*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2]),
        ) * a_c(2 * r_inner, 2 * r_part) * (C[i + 6*m] - C[i + 14*m]) + rho_cat*r4.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])

        dCdt[i + 15*m] = (1/(rho_cat * 450))*(h(
            rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            Molecule.mu_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0),
            2*r_part,
            Molecule.kappa_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            Molecule.Cp_gas_mix(
                C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            )
        ) * a_c(2*r_inner, 2*r_part) * (C[i + 7*m] - C[i + 15*m]) + ((1-porosity(2*r_inner, 2*r_part))*rho_cat*((-r1.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])*r1.H_rxn(C[i + 15*m])) + (-r2.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])*r2.H_rxn(C[i + 15*m])) + (-r3.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])*r3.H_rxn(C[i + 15*m])) + (-r4.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])*r4.H_rxn(C[i + 15*m])) + (-r5.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])*r5.H_rxn(C[i + 15*m])))))

    return dCdt


uinit = np.zeros(num_vars*m)
uinit[0] = C_A0
uinit[m] = C_B0
uinit[8*m:9*m] = 1e-10
uinit[m*7:m*8] = T_0
uinit[m*15:m*16] = T_0

time = np.linspace(0, t_dur, snaps)

start = tm.time()
sol = solve_ivp(deriv, (0, t_dur), uinit, method='RK23', t_eval=time, atol=1e-4, rtol=1e-4)
stop = tm.time()

print(stop - start)

x_points = np.linspace(0, length, m)

plt.plot(x_points, sol.y[:m, -1], x_points, sol.y[m:2*m, -1], x_points, sol.y[2*m:3*m, -1], x_points, sol.y[3*m:4*m, -1], x_points, sol.y[4*m:5*m, -1], x_points, sol.y[5*m:6*m, -1], x_points, sol.y[6*m:7*m, -1])

plt.show()


plt.plot(time, sol.y[16*m-1])
plt.show()

plt.plot(x_points, sol.y[7*m:8*m, -1], x_points, sol.y[15*m:16*m, -1])
plt.show()

print(sol.y[m - 1, -1])

# T, X = np.meshgrid(time, x_points)
# Z = sol.y[:m, :]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(T, X, Z, cmap='viridis')
# ax.set_xlabel('Time')
# ax.set_ylabel('Space')
# ax.set_zlabel('Concentration')
# plt.show()
