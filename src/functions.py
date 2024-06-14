from data import *
from math import tanh
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1000)
def rho_mix(T, P, C_tot, C_meth, C_O2, C_HCHO, C_H2O, C_CO, C_DME, C_DMM, C_N2):
    p_constant = P/(R*T*C_tot)
    return p_constant*(32.042e-3*C_meth + 32e-3*C_O2 + 28e-3*C_N2 + 18e-3*C_H2O + 30e-3*C_HCHO + 28e-3*C_CO + 46.047e-3*C_DME + 76.097e-3*C_DMM)


def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


por = porosity(2*r_inner, 2*r_part)


def a_c(dt, dp):  # fogler sida -> 698
    return 6*(1-por)/dp


def A_c(r):
    return 3.14*(r**2)


def u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I):
    return u_0 * rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0) / rho_mix(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I)


def q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I):
    return u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I)*A_c(r_inner)


def G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T):
    return rho_mix(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I)*u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I)


def C(F, F_T, P, T):
    return F/q_dot(F_T, P, T)


def F(T, P, C, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T):
    return C*q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I)


def Re(rho, mu, U, dp):
    return U*rho*dp/mu


def Pr(mu, kappa, Cp):
    return mu*Cp/kappa


def Sc(rho, mu, D_eff):
    return (mu/rho)/D_eff


def Sh(rho, mu, U, dp, D_eff):
    return 2 + 0.6*(Re(rho, mu, U, dp)**0.5)*(Sc(rho, mu, D_eff)**(1/3))


def Nu(rho, mu, U, dp, kappa, Cp):
    return 2 + 0.6*(Re(rho, mu, U, dp)**0.5)*(Pr(mu, kappa, Cp)**(1/3))


def k_c(rho, mu, U, dp, D_eff):
    return D_eff*Sh(rho, mu, U, dp, D_eff)/(dp*por)


def h(rho, mu, U, dp, kappa, Cp):
    return kappa*Nu(rho, mu, U, dp, kappa, Cp)/(dp*por)

def theta(k, De, C, T, dpe=Dpe, rho_c=rho_cat, e1=0, e2=0):
    if C < 0:
        C = 0

    thiele_root = (rho_c*k*((R*T*9.8e-6)**e2)*(C**e1)/(0.106*De))**0.5
    thiele = dpe * thiele_root

    return thiele

def eta(t_mod):
    return tanh(t_mod)/(t_mod + 1e-6)

def mass_mat(size, start_ind):
    M = np.eye(size)
    for i in range(start_ind, size):
        M[i, i] = 0

    return M


def reactor_len(w):
    return w / (rho_cat*(1-por)*A_c(r_inner))
