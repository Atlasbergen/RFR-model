from data import *
from math import tanh
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1000)
def rho_mix(T, P, C_tot, C_meth, C_O2, C_HCHO, C_H2O, C_CO, C_DME, C_DMM, C_N2):
    p_constant = P/(R*T*C_tot)
    return p_constant*(32.042e-3*C_meth + 32e-3*C_O2 + 28e-3*C_N2 + 18e-3*C_H2O + 30e-3*C_HCHO + 28e-3*C_CO + 46.047e-3*C_DME + 76.097e-3*C_DMM)

rho_noll = rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0)

def reactor_len(w):
    return w / (rho_cat*(1-por)*A_c(r_inner))

def reactor_weight(len):
    return len*(rho_cat*(1-por)*A_c(r_inner))

def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


por = porosity(2*r_inner, Dpe)

u_int = u_0/por

def a_c(dt, dp):  # fogler sida -> 698
    return 6*(1-por)/dp

def A_c(r):
    return 3.14*(r**2)


def u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I):
    return u_0 * rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0) / rho_mix(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I)

def u2(u_0, T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I):
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
    return U*rho*dp/(mu*(1-por))


def Re2(rho, mu, U, dp):
    return U*rho*dp/(mu)

def Pr(mu, kappa, Cp):
    return mu*Cp/kappa


def Sc(rho, mu, D_eff):
    return (mu/rho)/D_eff


def Sh(rho, mu, U, dp, D_eff):
    return 1.17*(Re2(rho, mu, U, dp)**0.585)*(Sc(rho, mu, D_eff)**(1/3))


# def Nu(rho, mu, U, dp, kappa, Cp):
#     return (Re(rho, mu, U, dp)**0.5)*(Pr(mu, kappa, Cp)**(1/3))

def Nu(rho, mu, U, dp, kappa, Cp):
    return ((1.18*Re2(rho, mu, U, dp)**0.58)**4 + (0.23*Re(rho, mu, U, dp)**0.75))**0.25

def k_c(rho, mu, U, dp, D_eff):
    return D_eff*Sh(rho, mu, U, dp, D_eff)/(dp*por)


def h(rho, mu, U, dp, kappa, Cp):
    return kappa*Nu(rho, mu, U, dp, kappa, Cp)/(dp*por)
 
def theta(k, De, C, T, dpe=Dpe, rho_c=rho_cat, e1=0, e2=0, De_fac=0.106):
    if C < 0:
        C = 0

    try:
        thiele_root = (rho_c*k*((R*T*9.8e-6)**e2)*(abs(C)**e1)/(De_fac*De))**0.5
        thiele = dpe * thiele_root
    except (ZeroDivisionError, OverflowError):
        thiele = 0
    
    return thiele

def eta(t_mod):
    try:
        x = tanh(t_mod)/(t_mod + 1e-6)            
    except:
        x = 0
        
    return x
    

def mass_mat(size, start_ind):
    M = np.eye(size)
    for i in range(start_ind, size):
        M[i, i] = 0

    return M

def thermcond(A, B, C, D, T): # transport properties of hydrocarbons
    return A  + (B/T) + (C*T) + (D*(T**2))

def thermcond_cat(T):
    return 0.667*10**(thermcond(2.1765, 3.1215E+00, -1.6460E-04, 2.6475E-08, T)) + 0.333*10**(thermcond(1.9434, 1.9215E+01, -2.9286E-04, -1.7150E-07, T))

def Cp_cat(T): # nist
    return 1.16234287e+02 + 8.48538599e-01*T + (-7.35380878e-04*(T**2))

# print(thermcond_cat(500))

def Pe(re, sc):
    return (0.54/(1 + (9.2/(re*sc))))**-1

def Pe_T(re, pr):
    return (0.54/(1 + (9.2/(re*pr))))**-1

def D_ax(dp, u, re, sc):
    return dp*u*Pe(re, sc)

def K_ax(dp, Cp, re, pr):
    return dp*u_0*rho_noll*Cp*Pe_T(re, pr)
