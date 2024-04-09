import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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


def mu_mix(T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, F_T):
    return (F_A/F_T)*CH3OH.mu(T) + (F_B/F_T)*O2.mu(T) + (F_C/F_T)*HCHO.mu(T) + (F_D/F_T)*H2O.mu(T) + (F_E/F_T)*CO.mu(T) + (F_F/F_T)*DME.mu(T) + (F_G/F_T)*DMM.mu(T) + (F_I/F_T)*N2.mu(T)


def B_0_new(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r, d_t, d_p):
    return (G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r)*((1-porosity(d_t, d_p))/(rho_mix(T_0, P_0, F_T0, F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, F_I0)*d_p*(porosity(d_t, d_p)**3)))*(((150*(1-porosity(d_t, d_p))*mu_mix(T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, F_T))/d_p) + 1.75*G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r)))


def dFdw(w, F):
    F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T = F
    F_T = F_A + F_B + F_C + F_D + F_E + F_F + F_G + F_I

    return [
        -(r1.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + 2*r3.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + 2*r4.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)),
        -0.5*(r1.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + r2.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + r5.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)),
        (r1.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + r5.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)) - (r2.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + r4.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)),
        r1.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + r2.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + r3.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + r4.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) + r5.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T),
        r2.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T),
        r3.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T) - 0.5*r5.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T),
        r4.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T),
        0,
        -(B_0_new(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r_inner, 2*r_inner, 2*r_part)/(A_c(r_inner)*(1-porosity(2*r_inner, 2*r_part))*rho_cat))*(T/T_0)*(P_0/P)*(F_T/F_T0)*0.0000098692,
        ((-r1.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)*r1.H_rxn(T))+(-r2.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)*r2.H_rxn(T))+(-r3.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)*r3.H_rxn(T))+(-r4.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)*r4.H_rxn(T))+(-r5.r(T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T)*r5.H_rxn(T)))/(F_A*CH3OH.Cp(T) + F_B*O2.Cp(T) + F_C*HCHO.Cp(T) + F_D*H2O.Cp(T) + F_E*CO.Cp(T) + F_F*DME.Cp(T) + F_G*DMM.Cp(T) + F_I*N2.Cp(T)),
    ]


w_span = (0, w_cat)
w_eval = np.linspace(0, w_cat, 1000)
F_0 = [F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, F_I0, P_0, T_0]

sol = solve_ivp(dFdw, w_span, F_0, method='RK45', t_eval=w_eval, rtol=1e-6, atol=1e-9)


w = sol.t
Y_T = sol.y[9]
Y_P = sol.y[8]
Y_A = sol.y[0]
Y_B = sol.y[1]
Y_C = sol.y[2]
Y_D = sol.y[3]
Y_E = sol.y[4]
Y_F = sol.y[5]
Y_G = sol.y[6]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

ax1.plot(w, Y_A, w, Y_B, w, Y_C, w, Y_D, w, Y_E, w, Y_F, w, Y_G, linewidth=0.9)
ax1.tick_params(axis="both",direction="in")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlabel("catalyst weight, W [kg]")
ax1.set_ylabel("F [mol/s]")
ax1.legend([r"$F_{CH_3OH}$", r"$F_{O_2}$", r"$F_{HCHO}$", r"$F_{H_2O}$", r"$F_{CO}$", r"$F_{DME}$", r"$F_{DMM}$"], loc="center left")
ax1.grid(color='0.8')

ax2.plot(w, (P_0-Y_P), linewidth=0.9)
ax2.tick_params(axis="both",direction="in")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlabel("catalyst weight, W [kg]")
ax2.set_ylabel("Pressure drop [atm]")
ax2.grid(color='0.8')

ax3.plot(w, Y_T, linewidth=0.9)
ax3.tick_params(axis="both",direction="in")
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_xlabel("catalyst weight, W [kg]")
ax3.set_ylabel("Temperature [K]")
ax3.grid(color='0.8')

plt.show()

print((Y_A[0]-Y_A[-1])/Y_A[0])
print(Y_B)
