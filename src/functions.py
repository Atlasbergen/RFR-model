from data import *
import numpy as np


def rho_mix(T, P, C_tot, C_meth, C_O2, C_HCHO, C_H2O, C_CO, C_DME, C_DMM, C_N2):
    return 32.042e-3*(C_meth*P/(R*T*C_tot)) + 32e-3*(C_O2*P/(R*T*C_tot)) + 28e-3*(C_N2*P/(R*T*C_tot)) + 18e-3*(C_H2O*P/(R*T*C_tot)) + 30e-3*(C_HCHO*P/(R*T*C_tot)) + 28e-3*(C_CO*P/(R*T*C_tot)) + 46.047e-3*(C_DME*P/(R*T*C_tot)) + 76.097e-3*(C_DMM*P/(R*T*C_tot))


def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


def a_c(dt, dp):  # fogler sida -> 698
    return 6*(1-porosity(dt, dp))/dp


def A_c(r):
    return 3.14*(r**2)


# def q_dot(F_T, P, T):
#     return F_T*R*T/P
#
#
# def u(F_T, P, T, r):
#     return q_dot(F_T, P, T) / (3.14*(r**2))


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
    return D_eff*Sh(rho, mu, U, dp, D_eff)/(dp*porosity(2*r_inner, 2*r_part))


def h(rho, mu, U, dp, kappa, Cp):
    return kappa*Nu(rho, mu, U, dp, kappa, Cp)/(dp*porosity(2*r_inner, 2*r_part))


def mass_mat(size, start_ind):
    M = np.eye(size)
    for i in range(start_ind, size):
        M[i, i] = 0

    return M


def reactor_len(w):
    return w / (rho_cat*(1-porosity(2*r_inner, 2*r_part))*A_c(r_inner))

if __name__ == "__main__":
    # test things here
    print(1-porosity(2*r_inner, 2*r_part), porosity(2*r_inner, 2*r_part))
    print((1-porosity(2*r_inner, 2*r_part))/porosity(2*r_inner, 2*r_part))
