from data import *
from scipy.integrate import quad
import numpy as np


def mu(T, C1, C2, C3):  # Perry 2-267
    return C1*(T**C2)/(1 + (C3/T))


def rho_mix(T, P, F_tot, F_meth, F_HCHO, F_DME, F_DMM, F_CO, F_H2O, F_O2, F_N2):
    return 32.042e-3*(F_meth*P*101325/(R*T*F_tot)) + 32e-3*(F_O2*P*101325/(R*T*F_tot)) + 28e-3*(F_N2*P*101325/(R*T*F_tot)) + 18e-3*(F_H2O*P*101325/(R*T*F_tot)) + 30e-3*(F_HCHO*P*101325/(R*T*F_tot)) + 28e-3*(F_CO*P*101325/(R*T*F_tot)) + 46.047e-3*(F_DME*P*101325/(R*T*F_tot)) + 76.097e-3*(F_DMM*P*101325/(R*T*F_tot))


def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


# the heat capacity functions will be removed in favor of the method in the Molecule class (The same for H_rxn)
# def C_p(T, C1, C2, C3, C4, C5):  # perry 2-149
#     return (C1 + (C2*(C3/(T*np.sinh(C3/T)))**2) + (C4*(C5/(T*np.cosh(C5/T)))**2))*1e-3


def A_c(r):
    return 3.14*(r**2)


def q_dot(F_T, P, T):
    return F_T*R*T/(P*101325)


def u(F_T, P, T, r):
    return q_dot(F_T, P, T) / (3.14*(r**2))


def G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r):
    return rho_mix(T, P, F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I)*u(F_T, P, T, r)


def B_0(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r, d_t, d_p):
    return (G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r)*((1-porosity(d_t, d_p))/(rho_mix(T_0, P_0, F_T0, F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, F_I0)*d_p*(porosity(d_t, d_p)**3)))*(((150*(1-porosity(d_t, d_p))*mu(T, 6.5592E-07, 0.6081 , 54.714))/d_p) + 1.75*G(F_T0, F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, F_I0, P_0, T_0, r)))


if __name__ == "__main__":
    # test things here
    print(mu(300, 3.0663E-07, 0.69655, 205))
