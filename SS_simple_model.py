import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

R = 8.314  # [J/(mol*K)]
r_inner = 0.0025  # [m]
w_cat = 0.00011  # [kg]
r_part = 1.5e-6  # [m]
T_0 = 280 + 273.15  # [K]
F_A0 = 1e-6  # [mol/s]
P_A0 = 0.1  # [atm]

A_CH3OH = 2.6e-4  # [atm]
A_O2 = 1.423e-5  # [atm^0.5]
A_H2O = 5.5e-7  # [atm]
A_HCHO = 1.5e7  # [mol/(kg_cat s)]

Ea_CH3OH = -56780  # [J/mol]
Ea_O2 = -60320  # [J/mol]
Ea_H2O = -86450  # [J/mol]
Ea_HCHO = 86000  # [J/mol]

y_A0 = 0.1
eps = y_A0 * 0.5  # 0.5 from: CH3OH + 0.5O2 -> HCHO + H2O


# Arrhenius equation. Used for rate constant and equilibrium constants (based on kinetics from Deshmuk).
def k(A: float, T: float, Ea: float) -> float:
    return A * np.exp(-Ea / (R * T))


# Rate law (no pressure drop, isothermal, only main reaction)
def r(P: float, T: float, X: float) -> float:
    return (
        k(A_HCHO, T, Ea_HCHO)
        * (
            (k(A_CH3OH, T, Ea_CH3OH) * (P * ((1 - X) / (1 + eps * X))))
            / (
                1
                + k(A_CH3OH, T, Ea_CH3OH) * (P * ((1 - X) / (1 + eps * X)))
                + k(A_H2O, T, Ea_H2O) * (P * X / (1 + eps * X))
            )
        )
        * (
            (k(A_O2, T, Ea_O2) * abs((P * ((1 - 0.5 * X) / (1 + eps * X)))) ** 0.5)
            / (1 + k(A_O2, T, Ea_O2) * abs((P * ((1 - 0.5 * X) / (1 + eps * X)))) ** 0.5)
        )
    )


# def event(t, y):
#     return y[0] - 0.99
#
#
# event.terminal = True


def dXdw(w, X):
    return r(P_A0, T_0, X) / F_A0


w_span = (0, w_cat)
X_0 = [0]

sol = solve_ivp(
    dXdw,
    w_span,
    X_0,
    method="RK45",
    t_eval=np.linspace(w_span[0], w_span[1], 1000),
    # events=event
)

w = sol.t
Y = sol.y[0]
