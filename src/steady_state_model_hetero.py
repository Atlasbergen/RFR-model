import matplotlib.pyplot as plt
from diffeqpy import de

from classes import *
from functions import *

CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2 = [
    Molecule(
        "Methanol",
        Mw_Me,
        H_f_Me,
        Tb_Me,
        Vb_Me,
        Param_Mu_Me,
        Param_Cp_Me,
        Param_kappa_Me,
    ),
    Molecule(
        "Oxygen", 
        Mw_O2,
        H_f_O2,
        Tb_O2, 
        Vb_O2, 
        Param_Mu_O2, 
        Param_Cp_O2, 
        Param_kappa_O2
    ),
    Molecule(
        "Formaldehyde",
        Mw_HCHO,
        H_f_HCHO,
        Tb_HCHO,
        Vb_HCHO,
        Param_Mu_HCHO,
        Param_Cp_HCHO,
        Param_kappa_HCHO,
    ),
    Molecule(
        "Water",
        Mw_H2O,
        H_f_H2O,
        Tb_H2O,
        Vb_H2O,
        Param_Mu_H2O,
        Param_Cp_H2O,
        Param_kappa_H2O,
    ),
    Molecule(
        "Carbon Monoxide",
        Mw_CO,
        H_f_CO,
        Tb_CO,
        Vb_CO,
        Param_Mu_CO,
        Param_Cp_CO,
        Param_kappa_CO,
    ),
    Molecule(
        "DME",
        Mw_DME,
        H_f_DME,
        Tb_DME,
        Vb_DME,
        Param_Mu_DME,
        Param_Cp_DME,
        Param_kappa_DME,
    ),
    Molecule(
        "DMM",
        Mw_DMM,
        H_f_DMM,
        Tb_DMM,
        Vb_DMM,
        Param_Mu_DMM,
        [0, 0, 0, 0],
        Param_kappa_DMM,
    ),
    Molecule(
        "Nitrogen",
        Mw_N2,
        H_f_N2,
        Tb_N2,
        Vb_N2,
        Param_Mu_N2,
        Param_Cp_N2,
        Param_kappa_N2,
    ),
]


r1, r2, r3, r4, r5 = [
    Reaction("reaction_1", [1, 0.5], [1, 1], [CH3OH, O2], [HCHO, H2O]),
    Reaction("reaction_2", [1, 0.5], [1, 1], [HCHO, O2], [CO, H2O]),
    Reaction("reaction_3", [2], [1, 1], [CH3OH], [DME, H2O]),
    Reaction("reaction_4", [2, 1], [1, 1], [CH3OH, HCHO], [DMM, H2O]),
    Reaction("reaction_5", [1, 1], [2, 1], [DME, O2], [HCHO, H2O]),
]


def B_0(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T, r, d_t, d_p):
    return (
        G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T, r)
        * (
            (1 - porosity(d_t, d_p))
            / (
                rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0,
                        C_D0, C_E0, C_F0, C_G0, C_I0)
                * d_p
                * (porosity(d_t, d_p) ** 3)
            )
        )
        * (
            (
                (
                    150
                    * (1 - porosity(d_t, d_p))
                    * Molecule.mu_gas_mix(
                        T,
                        C_T,
                        [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I],
                        [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2],
                    )
                )
                / d_p
            )
            + 1.75 * G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T, r)
        )
    )


def f(df, f, p, t):
    C_A, C_B, C_C, C_D, C_E, C_F, C_G, P, T_f, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs, T_s = f
    C_T = C_A + C_B + C_C + C_D + C_E + C_F + C_G + C_I0

    df[0] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            CH3OH.D_eff(T_f, P, C_T, C_A, [C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_As - C_A)
    )
    df[1] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            O2.D_eff(T_f, P, C_T, C_B, [C_A, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Bs - C_B)
    )
    df[2] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            HCHO.D_eff(T_f, P, C_T, C_C, [C_B, C_A, C_D, C_E, C_F, C_G, C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Cs - C_C)
    )
    df[3] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            H2O.D_eff(T_f, P, C_T, C_D, [C_B, C_C, C_A, C_E, C_F, C_G, C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Ds - C_D)
    )
    df[4] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            CO.D_eff(T_f, P, C_T, C_E, [C_A, C_B, C_C, C_D, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Es - C_E)
    )
    df[5] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            DME.D_eff(T_f, P, C_T, C_F, [C_A, C_B, C_C, C_D, C_E, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Fs - C_F)
    )
    df[6] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            DMM.D_eff(T_f, P, C_T, C_G, [C_A, C_B, C_C, C_D, C_E, C_F, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Gs - C_G)
    )
    
    df[7] = -B_0(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0, P, T_f, r_inner, 2*r_inner, 2*r_part)*(T_f/T_0)*(P_0/P)*((C_T*u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))/(C_T0*u_0))
    df[8] = (
        h(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2*r_part,
            Molecule.kappa_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            Molecule.Cp_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            )
        )
        * a_c(2*r_inner, 2*r_part)
        * (1/(u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0)*(CH3OH.Cp(T_f)*C_A + O2.Cp(T_f)*C_B + HCHO.Cp(T_f)*C_C + H2O.Cp(T_f)*C_D + CO.Cp(T_f)*C_E + DME.Cp(T_f)*C_F + DMM.Cp(T_f)*C_G + N2.Cp(T_f)*C_I0)))
        * (T_s - T_f)
    )

    df[9] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        Molecule.mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        CH3OH.D_eff(T_f, P, C_T, C_A, [C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_A - C_As) - (
        rho_cat * (1-porosity(2*r_inner, 2*r_part)) * (r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + 2*r3.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + 2*r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[10] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        Molecule.mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        O2.D_eff(T_f, P, C_T, C_B, [C_A, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_B - C_Bs) - (
        0.5 * rho_cat * (1-porosity(2*r_inner, 2*r_part)) * (r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r5.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[11] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        Molecule.mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        HCHO.D_eff(T_f, P, C_T, C_C, [C_B, C_A, C_D, C_E, C_F, C_G, C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_C - C_Cs) + (
        rho_cat * (1-porosity(2*r_inner, 2*r_part)) * (r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r5.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[12] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        Molecule.mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        H2O.D_eff(T_f, P, C_T, C_D, [C_B, C_C, C_A, C_E, C_F, C_G, C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_D - C_Ds) + (
        rho_cat * (1-porosity(2*r_inner, 2*r_part)) * (r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, 0, 0, 0) + r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r3.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + 0.5*r5.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[13] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        Molecule.mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        CO.D_eff(T_f, P, C_T, C_E, [C_A, C_B, C_C, C_D, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_E - C_Es) + (
        rho_cat * (1-porosity(2*r_inner, 2*r_part)) * r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)
    )
    df[14] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        Molecule.mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        DME.D_eff(T_f, P, C_T, C_F, [C_A, C_B, C_C, C_D, C_E, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_F - C_Fs) + (
        rho_cat * (1-porosity(2*r_inner, 2*r_part)) * (r3.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - 0.5*r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[15] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        Molecule.mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        DMM.D_eff(T_f, P, C_T, C_G, [C_A, C_B, C_C, C_D, C_E, C_F, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_G - C_Gs) + (
        rho_cat * (1-porosity(2*r_inner, 2*r_part)) * r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)
    )
    df[16] = (
        (h(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            Molecule.mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2*r_part,
            Molecule.kappa_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            Molecule.Cp_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            )
        )
        * a_c(2*r_inner, 2*r_part) * (T_f - T_s)) + ((1-porosity(2*r_inner, 2*r_part))*rho_cat*((-r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r1.H_rxn(T_s)) + (-r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r2.H_rxn(T_s)) + (-r3.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r3.H_rxn(T_s)) + (-r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r4.H_rxn(T_s)) + (-r5.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r5.H_rxn(T_s))))
    )

    return df


def condition(out, u, t, integrator):
    out[0] = u[0]
    out[1] = u[1]
    out[2] = u[2]
    out[3] = u[3]
    out[4] = u[4]
    out[5] = u[5]
    out[6] = u[6]
    out[7] = u[7] - 1000
    out[8] = u[8]  # - 700
    out[9] = u[9]
    out[10] = u[10]
    out[11] = u[11]
    out[12] = u[12]
    out[13] = u[13]
    out[14] = u[14]
    out[15] = u[15]
    out[16] = u[16]
    return out


def affect_b(integrator, idx):
    if idx == 1:
        integrator.u[0] = 0
    elif idx == 2:
        integrator.u[1] = 0
    elif idx == 3:
        integrator.u[2] = 0
    elif idx == 4:
        integrator.u[3] = 0
    elif idx == 5:
        integrator.u[4] = 0
    elif idx == 6:
        integrator.u[5] = 0
    elif idx == 7:
        integrator.u[6] = 0
    elif idx == 8:
        de.terminate_b(integrator)
    elif idx == 9:
        de.terminate_b(integrator)
    elif idx == 10:
        integrator.u[9] = 0
    elif idx == 11:
        integrator.u[10] = 0
    elif idx == 12:
        integrator.u[11] = 0
    elif idx == 13:
        integrator.u[12] = 0
    elif idx == 14:
        integrator.u[13] = 0
    elif idx == 15:
        integrator.u[14] = 0
    elif idx == 16:
        integrator.u[15] = 0
    elif idx == 17:
        integrator.u[16] = 0


cb = de.VectorContinuousCallback(condition, affect_b, 17)

f0 = [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, P_0, T_0, C_As0, C_Bs0, C_Cs0, C_Ds0, C_Es0, C_Fs0, C_Gs0, T_0]

M = mass_mat(17, 9)
z_span = (0, reactor_len(w_cat))


my_func = de.ODEFunction(f, mass_matrix=M)
prob_mm = de.ODEProblem(my_func, f0, z_span)
sol = de.solve(
    prob_mm,
    de.Rodas5(autodiff=False),
    callback=cb,
    saveat=0.01,
    reltol=1e-6,
    abstol=1e-6,
)

z = sol.t
u_vals = np.array([sol(i) for i in z]).T

u_A = u_vals[0] 
u_B = u_vals[1] 
u_C = u_vals[2] 
u_D = u_vals[3]
u_E = u_vals[4]
u_F = u_vals[5]
u_G = u_vals[6]
u_P = u_vals[7]
u_T = u_vals[8]
u_As = u_vals[9]
u_Bs = u_vals[10]
u_Cs = u_vals[11]
u_Ds = u_vals[12]
u_Es = u_vals[13]
u_Fs = u_vals[14]
u_Gs = u_vals[15]
u_Ts = u_vals[16]


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 18))

ax1.plot(z, u_A, z, u_B, z, u_C, z, u_D, z, u_E, z, u_F, z, u_G, linewidth=0.8)
ax1.set_title("Molarflow in bulk vs reactor length")
ax1.tick_params(axis="both",direction="in")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlabel("Reactor length, z [m]")
ax1.set_ylabel("Molar flow, F [mol/s]")
ax1.legend([r"$F_{CH_3OH}$", r"$F_{O_2}$", r"$F_{HCHO}$", r"$F_{H_2O}$", r"$F_{CO}$", r"$F_{DME}$", r"$F_{DMM}$"], loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=7)
ax1.grid(color='0.8')

ax2.plot(z, u_As, z, u_Bs, z, u_Cs, z, u_Ds, z, u_Es, z, u_Fs, z, u_Gs, linewidth=0.8)
ax2.set_title("Concentration on catalyst surface vs reactor length")
ax2.tick_params(axis="both",direction="in")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlabel("Reactor length, z [m]")
ax2.set_ylabel(r"$C_{cat}$ [mol/m³]")
ax2.legend([r"$C_{CH_3OH}$", r"$C_{O_2}$", r"$C_{HCHO}$", r"$C_{H_2O}$", r"$C_{CO}$", r"$C_{DME}$", r"$C_{DMM}$"], loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=7)
ax2.grid(color='0.8')

ax3.plot(z, (P_0 - np.array(u_P))*1e2/P_0, linewidth=0.8)
ax3.tick_params(axis="both",direction="in")
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_xlabel("Reactor length, z [m]")
ax3.set_ylabel(r"Pressure drop, $\Delta P$ [%]")
ax3.grid(color='0.8')

ax4.plot(z, u_T, z, u_Ts, linewidth=0.8)
ax4.tick_params(axis="both",direction="in")
ax4.spines[["top", "right"]].set_visible(False)
ax4.set_xlabel("Reactor length, z [m]")
ax4.set_ylabel("Temperature [K]")
ax4.legend([r"$T_{f}$", r"$T_{s}$"])
ax4.grid(color='0.8')


plt.subplots_adjust(hspace=0.3)
plt.show()

print(f"\nReactor length = {reactor_len(w_cat)} m\n")

print(f"\nReactor entrance:\nF_CH₃OH = {sol(z[0])[0]} mol/s\nF_O₂ = {sol(z[0])[1]} mol/s\nF_HCHO = {sol(z[0])[2]} mol/s\nF_H₂O = {sol(z[0])[3]} mol/s\nF_CO = {sol(z[0])[4]} mol/s\nF_DME = {sol(z[0])[5]} mol/s\nF_DMM = {sol(z[0])[6]} mol/s\nCs_CH₃OH = {sol(z[0])[9]} mol/s\nCs_O₂ = {sol(z[0])[10]} mol/s\nCs_HCHO = {sol(z[0])[11]} mol/s\nCs_H₂O = {sol(z[0])[12]} mol/s\nCs_CO = {sol(z[0])[13]} mol/s\nCs_DME = {sol(z[0])[14]} mol/s\nCs_DMM = {sol(z[0])[15]} mol/s\nP = {sol(z[0])[7]} Pa\nT_f = {sol(z[0])[8]} K\nT_s = {sol(z[0])[16]} K\n")

print(f"\nReactor exit:\nF_CH₃OH = {sol(z[-1])[0]} mol/s\nF_O₂ = {sol(z[-1])[1]} mol/s\nF_HCHO = {sol(z[-1])[2]} mol/s\nF_H₂O = {sol(z[-1])[3]} mol/s\nF_CO = {sol(z[-1])[4]} mol/s\nF_DME = {sol(z[-1])[5]} mol/s\nF_DMM = {sol(z[-1])[6]} mol/s\nCs_CH₃OH = {sol(z[-1])[9]} mol/s\nCs_O₂ = {sol(z[-1])[10]} mol/s\nCs_HCHO = {sol(z[-1])[11]} mol/s\nCs_H₂O = {sol(z[-1])[12]} mol/s\nCs_CO = {sol(z[-1])[13]} mol/s\nCs_DME = {sol(z[-1])[14]} mol/s\nCs_DMM = {sol(z[-1])[15]} mol/s\nP = {sol(z[-1])[7]} Pa\nT_f = {sol(z[-1])[8]} K\nT_s = {sol(z[-1])[16]} K\n")

print(f"Conversion of Methanol: {round((u_A[0]-u_A[-1])*1e2/u_A[0], 4)}%")
