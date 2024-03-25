import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions import *


def dFdw(w, F):
    F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T = F
    F_T = F_A + F_B + F_C + F_D + F_E + F_F + F_G + F_I
    
    return [
        -r_1(F_A, F_B, F_D, F_T, P, T_0) - 2*r_3(F_A, F_D, F_F, F_T, P, T_0) - 2*r_4(F_A, F_C, F_D, F_G, F_T, P, T_0),
        -0.5*(r_1(F_A, F_B, F_D, F_T, P, T_0) + r_2(F_A, F_B, F_C, F_D, F_E, F_T, P, T_0) + r_5(F_B, F_F, F_T, P, T_0)),
        r_1(F_A, F_B, F_D, F_T, P, T_0) - r_2(F_A, F_B, F_C, F_D, F_E, F_T, P, T_0) - r_4(F_A, F_C, F_D, F_G, F_T, P, T_0) + r_5(F_B, F_F, F_T, P, T_0),
        r_1(F_A, F_B, F_D, F_T, P, T_0) + r_2(F_A, F_B, F_C, F_D, F_E, F_T, P, T_0) + r_3(F_A, F_D, F_F, F_T, P, T_0) + r_4(F_A, F_C, F_D, F_G, F_T, P, T_0) + 0.5*r_5(F_B, F_F, F_T, P, T_0),
        r_2(F_A, F_B, F_C, F_D, F_E, F_T, P, T_0),
        r_3(F_A, F_D, F_F, F_T, P, T_0) - 0.5*r_5(F_B, F_F, F_T, P, T_0),
        r_4(F_A, F_C, F_D, F_G, F_T, P, T_0),
        0,
        -(B_0(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r_inner, 2*r_inner, 2*r_part)/(A_c(r_inner)*(1-porosity(2*r_inner, 2*r_part))*rho_cat))*(T/T_0)*(P_0/P)*(F_T/F_T0)*0.0000098692,
        ((-r_1(F_A, F_B, F_D, F_T, P, T)*H_rxn_1(T))+(-r_2(F_A, F_B, F_C, F_D, F_E, F_T, P, T)*H_rxn_2(T))+(-r_3(F_A, F_D, F_F, F_T, P, T)*H_rxn_3(T))+(-r_4(F_A, F_C, F_D, F_G, F_T, P, T)*H_rxn_4(T))+(-r_5(F_B, F_F, F_T, P, T)*H_rxn_5(T)))/(F_A*C_p_Me(T) + F_B*C_p_O2(T) + F_C*C_p_HCHO(T) + F_D*C_p_H2O(T) + F_E*C_p_CO(T) + F_F*C_p_DME(T) + F_G*C_p_DMM(T) + F_I*C_p_N2(T)),
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

ax2.plot(w, 1-Y_P, linewidth=0.9)
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
