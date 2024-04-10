from diffeqpy import de
import matplotlib.pyplot as plt
from functions import *
from classes import *


CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2 = [
    Molecule("Methanol", Mw_Me, H_f_Me, Tb_Me, Vb_Me, Param_Mu_Me, Param_Cp_Me, Param_kappa_Me),
    Molecule("Oxygen", Mw_O2, H_f_O2, Tb_O2, Vb_O2, Param_Mu_O2, Param_Cp_O2, Param_kappa_O2),
    Molecule("Formaldehyde", Mw_HCHO, H_f_HCHO, Tb_HCHO, Vb_HCHO, Param_Mu_HCHO, Param_Cp_HCHO, Param_kappa_HCHO),
    Molecule("Water", Mw_H2O, H_f_H2O, Tb_H2O, Vb_H2O, Param_Mu_H2O, Param_Cp_H2O, Param_kappa_H2O),
    Molecule("Carbon Monoxide", Mw_CO, H_f_CO, Tb_CO, Vb_CO, Param_Mu_CO, Param_Cp_CO, Param_kappa_CO),
    Molecule("DME", Mw_DME, H_f_DME, Tb_DME, Vb_DME, Param_Mu_DME, Param_Cp_DME, Param_kappa_DME),
    Molecule("DMM", Mw_DMM, H_f_DMM, Tb_DMM, Vb_DMM, Param_Mu_DMM, [0, 0, 0, 0], Param_kappa_DMM),
    Molecule("Nitrogen", Mw_N2, H_f_N2, Tb_N2, Vb_N2, Param_Mu_N2, Param_Cp_N2, Param_kappa_N2),
]


r1, r2, r3, r4, r5 = [
    Reaction("reaction_1", [1, 0.5], [1, 1], [CH3OH, O2], [HCHO, H2O]),
    Reaction("reaction_2", [1, 0.5], [1, 1], [HCHO, O2], [CO, H2O]),
    Reaction("reaction_3", [2], [1, 1], [CH3OH], [DME, H2O]),
    Reaction("reaction_4", [2, 1], [1, 1], [CH3OH, HCHO], [DMM, H2O]),
    Reaction("reaction_5", [1, 1], [2, 1], [DME, O2], [HCHO, H2O]),
]


def B_0_new(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r, d_t, d_p):
    return (G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r)*((1-porosity(d_t, d_p))/(rho_mix(T_0, P_0, F_T0, F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, F_I0)*d_p*(porosity(d_t, d_p)**3)))*(((150*(1-porosity(d_t, d_p))*Molecule.mu_gas_mix(T, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]))/d_p) + 1.75*G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r)))


def f(df, f, p, t):
    F_A, F_B, F_C, F_D, C_As, C_Bs, C_Cs, C_Ds = f
    F_T = F_A + F_B + F_C + F_D + F_I0

    df[0] = k_c(rho_mix(T_0, P_0, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0), Molecule.mu_gas_mix(T_0, F_T, [F_A, F_B, F_C, F_D, F_I0], [CH3OH, O2, HCHO, H2O, N2]), u(F_T, P_0, T_0, r_inner), 2*r_part, CH3OH.D_eff(T_0, P_0, F_T, F_A, [F_B, F_C, F_D, F_I0], [O2, HCHO, H2O, N2]))*a_c(2*r_inner, 2*r_part)*A_c(r_inner)*(C_As - C(F_A, F_T, P_0, T_0))
    df[1] = k_c(rho_mix(T_0, P_0, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0), Molecule.mu_gas_mix(T_0, F_T, [F_A, F_B, F_C, F_D, F_I0], [CH3OH, O2, HCHO, H2O, N2]), u(F_T, P_0, T_0, r_inner), 2*r_part, O2.D_eff(T_0, P_0, F_T, F_B, [F_A, F_C, F_D, F_I0], [CH3OH, HCHO, H2O, N2]))*a_c(2*r_inner, 2*r_part)*A_c(r_inner)*(C_Bs - C(F_B, F_T, P_0, T_0))
    df[2] = k_c(rho_mix(T_0, P_0, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0), Molecule.mu_gas_mix(T_0, F_T, [F_A, F_B, F_C, F_D, F_I0], [CH3OH, O2, HCHO, H2O, N2]), u(F_T, P_0, T_0, r_inner), 2*r_part, HCHO.D_eff(T_0, P_0, F_T, F_C, [F_B, F_A, F_D, F_I0], [O2, CH3OH, H2O, N2]))*a_c(2*r_inner, 2*r_part)*A_c(r_inner)*(C_Cs - C(F_C, F_T, P_0, T_0))
    df[3] = k_c(rho_mix(T_0, P_0, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0), Molecule.mu_gas_mix(T_0, F_T, [F_A, F_B, F_C, F_D, F_I0], [CH3OH, O2, HCHO, H2O, N2]), u(F_T, P_0, T_0, r_inner), 2*r_part, H2O.D_eff(T_0, P_0, F_T, F_D, [F_B, F_C, F_A, F_I0], [O2, HCHO, CH3OH, N2]))*a_c(2*r_inner, 2*r_part)*A_c(r_inner)*(C_Ds - C(F_D, F_T, P_0, T_0))

    df[4] = k_c(rho_mix(T_0, P_0, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0), Molecule.mu_gas_mix(T_0, F_T, [F_A, F_B, F_C, F_D, F_I0], [CH3OH, O2, HCHO, H2O, N2]), u(F_T, P_0, T_0, r_inner), 2*r_part, CH3OH.D_eff(T_0, P_0, F_T, F_A, [F_B, F_C, F_D, F_I0], [O2, HCHO, H2O, N2]))*a_c(2*r_inner, 2*r_part)*(C(F_A, F_T, P_0, T_0) - C_As) + (-rho_cat*r1.r(T_0, C_As, C_Bs, C_Cs, C_Ds, 0, 0, 0))
    df[5] = k_c(rho_mix(T_0, P_0, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0), Molecule.mu_gas_mix(T_0, F_T, [F_A, F_B, F_C, F_D, F_I0], [CH3OH, O2, HCHO, H2O, N2]), u(F_T, P_0, T_0, r_inner), 2*r_part, O2.D_eff(T_0, P_0, F_T, F_B, [F_A, F_C, F_D, F_I0], [CH3OH, HCHO, H2O, N2]))*a_c(2*r_inner, 2*r_part)*(C(F_B, F_T, P_0, T_0) - C_Bs) + (-rho_cat*r1.r(T_0, C_As, C_Bs, C_Cs, C_Ds, 0, 0, 0))
    df[6] = k_c(rho_mix(T_0, P_0, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0), Molecule.mu_gas_mix(T_0, F_T, [F_A, F_B, F_C, F_D, F_I0], [CH3OH, O2, HCHO, H2O, N2]), u(F_T, P_0, T_0, r_inner), 2*r_part, HCHO.D_eff(T_0, P_0, F_T, F_C, [F_B, F_A, F_D, F_I0], [O2, CH3OH, H2O, N2]))*a_c(2*r_inner, 2*r_part)*(C(F_C, F_T, P_0, T_0) - C_Cs) + (rho_cat*r1.r(T_0, C_As, C_Bs, C_Cs, C_Ds, 0, 0, 0))
    df[7] = k_c(rho_mix(T_0, P_0, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0), Molecule.mu_gas_mix(T_0, F_T, [F_A, F_B, F_C, F_D, F_I0], [CH3OH, O2, HCHO, H2O, N2]), u(F_T, P_0, T_0, r_inner), 2*r_part, H2O.D_eff(T_0, P_0, F_T, F_D, [F_B, F_C, F_A, F_I0], [O2, HCHO, CH3OH, N2]))*a_c(2*r_inner, 2*r_part)*(C(F_D, F_T, P_0, T_0) - C_Ds) + (rho_cat*r1.r(T_0, C_As, C_Bs, C_Cs, C_Ds, 0, 0, 0))

    return df


f0 = [F_A0, F_B0, F_C0, F_D0, 0, 0, 0, 0]

M = mass_mat(8, 4)

z_span = (0, 3)
# diff_vars = [True, True, True, True, False, False, False, False]
# prob = de.DAEProblem(f, df0, f0, z_span, differential_vars=diff_vars)
# sol = de.solve(prob, de.ARKODE(), saveat=0.0001)

my_func = de.ODEFunction(f, mass_matrix = M)
prob_mm = de.ODEProblem(my_func, f0, z_span)
sol = de.solve(prob_mm, de.Rodas5(autodiff=False), saveat=0.001)

z = sol.t
u_vals = np.array([sol(i) for i in z]).T


plt.plot(z, u_vals[0], z, u_vals[1], z, u_vals[2], z, u_vals[3])
plt.show()
print(sol(z[-1]))
