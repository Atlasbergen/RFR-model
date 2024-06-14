import numpy as np
from scipy.integrate import solve_ivp
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
    Molecule("DMM", Mw_DMM, H_f_DMM, Tb_DMM, Vb_DMM, Param_Mu_DMM, Param_Cp_DMM, Param_kappa_DMM,),
    Molecule("Nitrogen", Mw_N2, H_f_N2, Tb_N2, Vb_N2, Param_Mu_N2, Param_Cp_N2, Param_kappa_N2,),
]

r1, r2, r3, r4, r5 = [
    Reaction("reaction_1", [1, 0.5], [1, 1], [CH3OH, O2], [HCHO, H2O]),
    Reaction("reaction_2", [1, 0.5], [1, 1], [HCHO, O2], [CO, H2O]),
    Reaction("reaction_3", [2], [1, 1], [CH3OH], [DME, H2O]),
    Reaction("reaction_4", [2, 1], [1, 1], [CH3OH, HCHO], [DMM, H2O]),
    Reaction("reaction_5", [1, 1], [2, 1], [DME, O2], [HCHO, H2O]),
]

rho_cat_p = rho_cat * (1-porosity(2*r_inner, 2*r_part))
AC = a_c(2 * r_inner, 2 * r_part)
part_dia = r_part * 2
eps_fac = (1-porosity(2*r_inner, 2*r_part))/porosity(2*r_inner, 2*r_part)
eps_fac_2 = porosity(2*r_inner, 2*r_part)/(1-porosity(2*r_inner, 2*r_part))

m = 60
snaps = 200
t_dur = 1000

length = reactor_len(w_cat)
dx = length / m

num_vars = 16 
entries = num_vars * m

def deriv(t, C):
    dCdt = np.zeros_like(C)
    # dCdt = [0] * entries

    for i in range(1, m):
        
        C_tot = C[i] + C[i + m] + C[i + 2*m] + C[i + 3*m] + C[i + 4*m] + C[i + 5*m] + C[i + 6*m] + C_I0
        u_all = u(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0)
        rho_gas_mix = rho_mix(C[i + 7*m], P_0, C_tot, C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0)
        mu_gas_all = mu_gas_mix(C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2])
        kappa_gas = kappa_gas_mix(C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2])
        Cp_gas = Cp_gas_mix(C[i + 7*m], C_tot, [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2])

        D_A_eff = CH3OH.D_eff(C[i + 7*m], P_0, C_tot, C[i], [C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2])
        D_B_eff = O2.D_eff(C[i + 7*m], P_0, C_tot, C[i + m], [C[i], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2])
        D_C_eff = HCHO.D_eff(C[i + 7*m], P_0, C_tot, C[i + 2*m], [C[i + m], C[i], C[i + 3*m], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2])
        D_D_eff = H2O.D_eff(C[i + 7*m], P_0, C_tot, C[i + 3*m], [C[i + m], C[i + 2*m], C[i], C[i + 4*m], C[i + 5*m], C[i + 6*m], C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2])
        D_E_eff = CO.D_eff(C[i + 7*m], P_0, C_tot, C[i + 4*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 5*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2])
        D_F_eff = DME.D_eff(C[i + 7*m], P_0, C_tot, C[i + 5*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 6*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2])
        D_G_eff = DMM.D_eff(C[i + 7*m], P_0, C_tot, C[i + 6*m], [C[i], C[i + m], C[i + 2*m], C[i + 3*m], C[i + 4*m], C[i + 5*m], C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2])


        n_1 = eta(
            theta(
                Reaction.k(A_HCHO, C[i + 15*m], Ea_HCHO),
                D_A_eff,
                C[i] + 1e-6,
                C[i + 15*m],
                e1=-1,
            )
        )

        n_2 = eta(
            theta(
                Reaction.k(A_CO, C[i + 15*m], Ea_CO),
                D_C_eff,
                1,
                C[i + 15*m],
                e2=1,
            )
        )

        n_3 = eta(
            theta(
                Reaction.k(A_DMEf, C[i + 15*m], Ea_DMEf),
                D_A_eff,
                1,
                C[i + 15*m],
                e2=1,
            )
        )

        n_4 = eta(
            theta(
                Reaction.k(A_DMMf, C[i + 15*m], Ea_DMMf),
                D_A_eff,
                C[i] + 1e-6,
                C[i + 15*m],
                e1=1,
                e2=2,
            )
        )

        n_5 = eta(
            theta(
                Reaction.k(A_DMEHCHO, C[i + 15*m], Ea_DMEHCHO),
                D_F_eff,
                C[i + 5*m] + 1e-6,
                C[i + 15*m],
                e1=-1,
            )
        )

        # print(theta(Reaction.k(A_HCHO, C[i + 15*m], Ea_HCHO), D_A_eff, C[i] + 1e-6, C[i + 15*m], e1=-1), n_1, n_2, n_3, n_4, n_5)
        
        r_1_all = n_1*r1.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])
        r_2_all = n_2*r2.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])
        r_3_all = n_3*r3.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])
        r_4_all = n_4*r4.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])
        r_5_all = n_5*r5.r(C[i + 15*m], C[i + 8*m], C[i + 9*m], C[i + 10*m], C[i + 11*m], C[i + 12*m], C[i + 13*m], C[i + 14*m])


        dCdt[i] = -u_all * ((C[i] - C[i-1]) / (dx)) + k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_A_eff,
        ) * AC * (C[i + 8*m] - C[i])

        dCdt[i + m] = -u_all * ((C[i + m] - C[i-1 + m]) / (dx)) + k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_B_eff,
        ) * AC * (C[i + 9*m] - C[i + m])

        dCdt[i + 2*m] = -u_all * ((C[i + 2*m] - C[i-1 + 2*m]) / (dx)) + k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_C_eff,
        ) * AC * (C[i + 10*m] - C[i + 2*m])

        dCdt[i + 3*m] = -u_all * ((C[i + 3*m] - C[i-1 + 3*m]) / (dx)) + k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_D_eff,
        ) * AC * (C[i + 11*m] - C[i + 3*m])
        
        dCdt[i + 4*m] = -u_all * ((C[i + 4*m] - C[i-1 + 4*m]) / (dx)) + k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_E_eff
        ) * AC * (C[i + 12*m] - C[i + 4*m])

        dCdt[i + 5*m] = -u_all * ((C[i + 5*m] - C[i-1 + 5*m]) / (dx)) + k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_F_eff,
        ) * AC * (C[i + 13*m] - C[i + 5*m])
        
        dCdt[i + 6*m] = -u_all * ((C[i + 6*m] - C[i-1 + 6*m]) / (dx)) + k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_G_eff,
        ) * AC * (C[i + 14*m] - C[i + 6*m])

        dCdt[i + 7*m] = -u_all * ((C[i + 7*m] - C[i-1 + 7*m]) / (dx)) + h(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            kappa_gas,
            Cp_gas
        ) * AC * (1/(CH3OH.Cp(C[i + 7*m])*C[i] + O2.Cp(C[i + 7*m])*C[i + m] + HCHO.Cp(C[i + 7*m])*C[i + 2*m] + H2O.Cp(C[i + 7*m])*C[i + 3*m] + CO.Cp(C[i + 7*m])*C[i + 4*m] + DME.Cp(C[i + 7*m])*C[i + 5*m] + DMM.Cp(C[i + 7*m])*C[i + 6*m] + N2.Cp(C[i + 7*m])*C_I0)) * (C[i + 15*m] - C[i + 7*m])
        
        dCdt[i + 8*m] = k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_A_eff,
        ) * AC * (C[i] - C[i + 8*m]) - rho_cat_p*eps_fac*(r_1_all + 2*r_3_all + 2*r_4_all)

        dCdt[i + 9*m] = k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_B_eff,
        ) * AC * (C[i + m] - C[i + 9*m]) - 0.5*rho_cat_p*eps_fac*(r_1_all + r_2_all + r_5_all)

        dCdt[i + 10*m] = k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_C_eff,
        ) * AC * (C[i + 2*m] - C[i + 10*m]) + rho_cat_p*eps_fac*((r_1_all + r_5_all) - (r_2_all + r_4_all))

        dCdt[i + 11*m] = k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_D_eff,
        ) * AC * (C[i + 3*m] - C[i + 11*m]) + rho_cat_p*eps_fac*(r_1_all + r_2_all + r_3_all + r_4_all + 0.5*r_5_all)

        dCdt[i + 12*m] = k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_E_eff,
    ) * AC * (C[i + 4*m] - C[i + 12*m]) + rho_cat_p*eps_fac*r_2_all

        dCdt[i + 13*m] = k_c(
            rho_gas_mix,
            mu_gas_all,
            u_all,
            part_dia,
            D_F_eff,
        ) * AC * (C[i + 5*m] - C[i + 13*m]) + rho_cat_p*eps_fac*(r_3_all - 0.5*r_5_all)
        
        dCdt[i + 14*m] = k_c(
        rho_gas_mix,
        mu_gas_all,
        u_all,
        part_dia,
        D_G_eff,
        ) * AC * (C[i + 6*m] - C[i + 14*m]) + rho_cat_p*eps_fac*r_4_all
        
        if i != m-1:
            dCdt[i + 15*m] = (1/(rho_cat_p * 374 * 0.8))*(((C[i + 15*m + 1] - 2*C[i + 15*m] + C[i + 15*m - 1])/dx**2) + eps_fac_2*h(
                rho_gas_mix,
                mu_gas_all,
                u_all,
                part_dia,
                kappa_gas,
                Cp_gas
            ) * AC * (C[i + 7*m] - C[i + 15*m]) + (rho_cat_p*((-r_1_all*r1.H_rxn(C[i + 15*m])) + (-r_2_all*r2.H_rxn(C[i + 15*m])) + (-r_3_all*r3.H_rxn(C[i + 15*m])) + (-r_4_all*r4.H_rxn(C[i + 15*m])) + (-r_5_all*r5.H_rxn(C[i + 15*m])))))
        else:
            dCdt[i + 15*m] = (1/(rho_cat_p * 374 * 0.8))*(eps_fac_2*h(
                rho_gas_mix,
                mu_gas_all,
                u_all,
                part_dia,
                kappa_gas,
                Cp_gas
            ) * AC * (C[i + 7*m] - C[i + 15*m]) + (rho_cat_p*((-r_1_all*r1.H_rxn(C[i + 15*m])) + (-r_2_all*r2.H_rxn(C[i + 15*m])) + (-r_3_all*r3.H_rxn(C[i + 15*m])) + (-r_4_all*r4.H_rxn(C[i + 15*m])) + (-r_5_all*r5.H_rxn(C[i + 15*m])))))

    return dCdt


uinit = np.zeros(num_vars*m)
uinit[0] = C_A0
uinit[m] = C_B0
uinit[8*m:9*m] = 1e-10
uinit[m*7:m*8] = T_0
uinit[m*15:m*16] = T_0

time = np.linspace(0, t_dur, snaps)

start = tm.time()
sol = solve_ivp(deriv, (0, t_dur), uinit, method='RK23', t_eval=time, atol=1e-6, rtol=1e-6)
stop = tm.time()

print(stop - start)

x_points = np.linspace(0, length, m)

np.savetxt("data/output_data.txt", sol.y)
