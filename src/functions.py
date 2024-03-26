from data import *
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np


def mu(T, C1, C2, C3):
    return C1*(T**C2)/(1 + (C3/T))


def rho_mix(T, P, F_tot, F_meth, F_HCHO, F_DME, F_DMM, F_CO, F_H2O, F_O2, F_N2):
    return 32.042e-3*(F_meth*P*101325/(R*T*F_tot)) + 32e-3*(F_O2*P*101325/(R*T*F_tot)) + 28e-3*(F_N2*P*101325/(R*T*F_tot)) + 18e-3*(F_H2O*P*101325/(R*T*F_tot)) + 30e-3*(F_HCHO*P*101325/(R*T*F_tot)) + 28e-3*(F_CO*P*101325/(R*T*F_tot)) + 46.047e-3*(F_DME*P*101325/(R*T*F_tot)) + 76.097e-3*(F_DMM*P*101325/(R*T*F_tot))


def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


def C_p(T, C1, C2, C3, C4, C5):
    return (C1 + (C2*(C3/(T*np.sinh(C3/T)))**2) + (C4*(C5/(T*np.cosh(C5/T)))**2))*1e-3


def C_p_HCHO(T):
    return C_p(T, 0.33503e5, 0.49394e5, 1.92800e3, 0.29728e5, 965.04)


def C_p_Me(T):
    return C_p(T, 0.39252e5, 0.87900e5, 1.91650e3, 0.53654e5, 896.7)


def C_p_H2O(T):
    return C_p(T, 0.33363e5, 0.26790e5, 2.61050e3, 0.08896e5, 1169)


def C_p_O2(T):
    return C_p(T, 0.29103e5, 0.10040e5, 2.52650e3, 0.09356e5, 1153.8)


def C_p_N2(T):
    return C_p(T, 0.29105e5, 0.08615e5, 1.70160e3, 0.00103e5, 909.79)


def C_p_CO(T):
    return C_p(T, 0.29108e5, 0.08773e5, 3.08510e3, 0.08455e5, 1538.2)


def C_p_DME(T):
    return C_p(T, 0.57431e5, 0.94494e5, 0.89551e3, 0.65065e5, 2467.4)


def C_p_DMM(T):
    return 51.161 + 0.16244*T + 8.26e-5*(T**2) + (-8.51e-8*(T**3))


def del_Cp_1(T):
    return C_p_HCHO(T) + C_p_H2O(T) - C_p_Me(T) - 0.5*C_p_O2(T)


def del_Cp_2(T):
    return C_p_CO(T) + C_p_H2O(T) - C_p_HCHO(T) - 0.5*C_p_O2(T)


def del_Cp_3(T):
    return C_p_DME(T) + C_p_H2O(T) - 2*C_p_Me(T)


def del_Cp_4(T):
    return C_p_DMM(T) + C_p_H2O(T) - 2*C_p_Me(T) - C_p_HCHO(T)


def del_Cp_5(T):
    return 2*C_p_HCHO(T) + C_p_H2O(T) - C_p_DME(T) - C_p_O2(T)


def H_rxn_1(T, T_ref=298.15):
    return H_rxn0_1 + quad(del_Cp_1, T_ref, T)[0]


def H_rxn_2(T, T_ref=298.15):
    return H_rxn0_2 + quad(del_Cp_2, T_ref, T)[0]


def H_rxn_3(T, T_ref=298.15):
    return H_rxn0_3 + quad(del_Cp_3, T_ref, T)[0]


def H_rxn_4(T, T_ref=298.15):
    return H_rxn0_4 + quad(del_Cp_4, T_ref, T)[0]


def H_rxn_5(T, T_ref=298.15):
    return H_rxn0_5 + quad(del_Cp_5, T_ref, T)[0]


def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


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


def K_eq_DME(T):
    return np.exp(-2.2158 + (2606.8 / T))


def K_eq_DMM(T):
    return np.exp(-20.416 + (9346.8 / T))


# Arrhenius equation. Used for rate constant and equilibrium constants (based on kinetics from Deshmuk).
def k(A: float, T: float, Ea: float) -> float:
    return A * np.exp(-Ea / (R * T))


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
