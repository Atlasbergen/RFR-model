import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from data import mu_air, rho_air


R = 8.314  # [J/(mol*K)]
r_inner = 0.0025  # [m]
w_cat = 0.00011  # [kg]
r_part = 150e-6  # [m]
T_0 = 280 + 273.15  # [K]
F_A0 = 1e-5  # [mol/s]
F_B0 = 0.5e-5  # [mol/S]
F_C0 = 0  # [mol/s]
F_D0 = 0  # [mol/s]
F_E0 = 0  # [mol/s]
F_F0 = 0  # [mol/s]
F_G0 = 0  # [mol/s]
F_I0 = 8e-5  # [mol/s]
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


def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


# function for crosssectional area
def A_c(r):
    return 3.14*(r**2)


# volumetric flow
def q_dot(F_T, P, T):
    return F_T*R*T/(P*101325)


# Superficial velocity
def u(F_T, P, T, r):
    return q_dot(F_T, P, T) / (3.14*(r**2))


# Superficial mass velocity
def G(F_T, P, T, r):
    return rho_air(T)*u(F_T, P, T, r)


# Beta zero parameter in ergun-equation
def B_0(F_T, P, T, r, d_t, d_p):
    return (G(F_T, P, T, r)*((1-porosity(d_t, d_p))/(rho_air(T)*d_p*(porosity(d_t, d_p)**3)))*(((150*(1-porosity(d_t, d_p))*mu_air(T))/d_p) + 1.75*G(F_T, P, T, r)))


# Equilibrium constant from deshmuk
def K_eq_DME(T):
    return np.exp(-2.2158 + (2606.8 / T))


# Equilibrium constant from deshmuk
def K_eq_DMM(T):
    return np.exp(-20.416 + (9346.8 / T))


# Arrhenius equation. Used for rate constant and equilibrium constants (based on kinetics from Deshmuk).
def k(A: float, T: float, Ea: float) -> float:
    return A * np.exp(-Ea / (R * T))


# reaction rate for first reaction
def r_1(F_A: float, F_B: float, F_D: float, F_T: float, P: float, T: float) -> float:
    return (
        k(A_HCHO, T, Ea_HCHO)
        * (
            (k(A_CH3OH, T, Ea_CH3OH) * (P * F_A / F_T))
            / (
                1
                + k(A_CH3OH, T, Ea_CH3OH) * (P * F_A / F_T)
                + k(A_H2O, T, Ea_H2O) * (P * F_D / F_T)
            )
        )
        * (
            (k(A_O2, T, Ea_O2) * (P * F_B / F_T) ** 0.5)
            / (1 + (k(A_O2, T, Ea_O2) * (P * F_B / F_T) ** 0.5))
        )
    )


def r_2(F_A: float, F_B: float, F_C: float, F_D: float, F_E: float, F_T: float, P: float, T: float) -> float:
    return (
        k(A_CO, T, Ea_CO)
        * (
            (P * F_C / F_T)
            / (
                1
                + k(A_CH3OH, T, Ea_CH3OH) * (P * F_A / F_T)
                + k(A_H2O, T, Ea_H2O) * (P * F_D / F_T)
            )
        )
        * (
            (k(A_O2, T, Ea_O2) * (P * F_B / F_T) ** 0.5)
            / (1 + (k(A_O2, T, Ea_O2) * (P * F_B / F_T) ** 0.5))
        )
    )


def r_3(F_A: float, F_D: float, F_F: float, F_T: float, P: float, T: float) -> float:
    return k(A_DMEf, T, Ea_DMEf) * (P * F_A / F_T) - (
        k(A_DMEf, T, Ea_DMEf) / K_eq_DME(T)
    ) * (P * F_F * F_D / (F_A * F_T))


def r_4(F_A: float, F_C: float, F_D: float, F_G: float, F_T: float, P: float, T: float) -> float:
    return k(A_DMMf, T, Ea_DMMf) * (P**2 * F_A * F_C/F_T**2) - (k(A_DMMf, T_0, Ea_DMMf)/K_eq_DMM(T))*(P*F_D*F_G/(F_A*F_T))


def r_5(F_B: float, F_F: float, F_T: float, P: float, T: float) -> float:
    return k(A_DMEHCHO, T, Ea_DMEHCHO)*(k(A_DME, T, Ea_DME)*(P*F_F/F_T)/(1 + k(A_DME, T, Ea_DME)*(P*F_F/F_T)))*((k(A_O2, T, Ea_O2) * (P * F_B / F_T) ** 0.5)/(1 + (k(A_O2, T, Ea_O2) * (P * F_B / F_T) ** 0.5)))


# setting up the system of differential equation
def dFdw(w, F):
    F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P = F
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
        -(B_0(F_T, P, T_0, r_inner, 2*r_inner, 2*r_part)/(A_c(r_inner)*(1-porosity(2*r_inner, 2*r_part))*700))*(P_0/P)*(F_T/F_T0)*0.0000098692
    ]


w_span = (0, w_cat)
w_eval = np.linspace(0, w_cat, 1000)
F_0 = [F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, F_I0, P_0]

sol = solve_ivp(dFdw, w_span, F_0, method='RK45', t_eval=w_eval, rtol=1e-6, atol=1e-9)


w = sol.t
Y_P = sol.y[8]
Y_A = sol.y[0]
Y_B = sol.y[1]
Y_C = sol.y[2]
Y_D = sol.y[3]
Y_E = sol.y[4]
Y_F = sol.y[5]
Y_G = sol.y[6]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(w, Y_A, w, Y_B, w, Y_C, w, Y_D, w, Y_E, w, Y_F, w, Y_G, linewidth=0.9)
ax1.tick_params(axis="both",direction="in")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlabel("catalyst weight, W [kg]")
ax1.set_ylabel("F [mol/s]")
ax1.legend([r"$F_{CH_3OH}$", r"$F_{O_2}$", r"$F_{HCHO}$", r"$F_{H_2O}$", r"$F_{CO}$", r"$F_{DME}$", r"$F_{DMM}$"], loc="upper left")
ax1.grid(color='0.8')

ax2.plot(w, Y_P, linewidth=0.9)
ax2.tick_params(axis="both",direction="in")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlabel("catalyst weight, W [kg]")
ax2.set_ylabel("Pressure [atm]")
ax2.grid(color='0.8')

plt.show()
