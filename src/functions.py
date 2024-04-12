from data import *
import numpy as np


def rho_mix(T, P, F_tot, F_meth, F_O2, F_HCHO, F_H2O, F_CO, F_DME, F_DMM, F_N2):
    return 32.042e-3*(F_meth*P/(R*T*F_tot)) + 32e-3*(F_O2*P/(R*T*F_tot)) + 28e-3*(F_N2*P/(R*T*F_tot)) + 18e-3*(F_H2O*P/(R*T*F_tot)) + 30e-3*(F_HCHO*P/(R*T*F_tot)) + 28e-3*(F_CO*P/(R*T*F_tot)) + 46.047e-3*(F_DME*P/(R*T*F_tot)) + 76.097e-3*(F_DMM*P/(R*T*F_tot))


def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


def a_c(dt, dp):  # fogler sida -> 698
    return 6*(1-porosity(dt, dp))/dp


def A_c(r):
    return 3.14*(r**2)


def q_dot(F_T, P, T):
    return F_T*R*T/P


def u(F_T, P, T, r):
    return q_dot(F_T, P, T) / (3.14*(r**2))


def G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r):
    return rho_mix(T, P, F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I)*u(F_T, P, T, r)


def B_0(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r, d_t, d_p):
    return (G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r)*((1-porosity(d_t, d_p))/(rho_mix(T_0, P_0, F_T0, F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, F_I0)*d_p*(porosity(d_t, d_p)**3)))*(((150*(1-porosity(d_t, d_p))*mu(T, 6.5592E-07, 0.6081 , 54.714))/d_p) + 1.75*G(F_T0, F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, F_I0, P_0, T_0, r)))


def C(F, F_T, P, T):
    return F/q_dot(F_T, P, T)


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
    return D_eff*Sh(rho, mu, U, dp, D_eff)/dp


def h(rho, mu, U, dp, kappa, Cp):
    return kappa*Nu(rho, mu, U, dp, kappa, Cp)/dp


def mass_mat(size, start_ind):
    M = np.eye(size)
    for i in range(start_ind, size):
        M[i, i] = 0

    return M


def reactor_len(w):
    return w / (rho_cat*(1-porosity(2*r_inner, 2*r_part))*A_c(r_inner))

if __name__ == "__main__":
    # test things here
    print(reactor_len(w_cat))
    print(u(F_T0, P_0, T_0, r_inner))
    print(C_A0)
