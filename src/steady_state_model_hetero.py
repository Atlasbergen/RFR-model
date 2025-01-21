import matplotlib.pyplot as plt
from diffeqpy import de
from classes import *
from functions import *

print("Reactor length: ", reactor_len(w_cat))
print("Reactor weight: ", reactor_weight(2))


CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2 = [
    Molecule("Methanol", Mw_Me, H_f_Me, Tb_Me, Vb_Me, Param_Mu_Me, Param_Cp_Me, Param_kappa_Me,),
    Molecule("Oxygen",  Mw_O2, H_f_O2,  Tb_O2, Vb_O2, Param_Mu_O2, Param_Cp_O2, Param_kappa_O2),
    Molecule("Formaldehyde", Mw_HCHO, H_f_HCHO, Tb_HCHO, Vb_HCHO, Param_Mu_HCHO, Param_Cp_HCHO, Param_kappa_HCHO,),
    Molecule("Water", Mw_H2O, H_f_H2O, Tb_H2O, Vb_H2O, Param_Mu_H2O, Param_Cp_H2O, Param_kappa_H2O,),
    Molecule("Carbon Monoxide", Mw_CO, H_f_CO, Tb_CO, Vb_CO, Param_Mu_CO, Param_Cp_CO, Param_kappa_CO,),
    Molecule("DME", Mw_DME, H_f_DME, Tb_DME, Vb_DME, Param_Mu_DME, Param_Cp_DME, Param_kappa_DME,),
    Molecule("DMM", Mw_DMM, H_f_DMM, Tb_DMM, Vb_DMM, Param_Mu_DMM, Param_Cp_DMM, Param_kappa_DMM,),
    Molecule("Nitrogen", Mw_N2, H_f_N2, Tb_N2, Vb_N2, Param_Mu_N2, Param_Cp_N2, Param_kappa_N2,),
]


r1, r2, r3, r4, r5 = [
    Reaction("reaction_1", [1, 0.5], [1, 1], [CH3OH, O2], [HCHO, H2O]),
    Reaction("reaction_2", [1, 0.5], [1, 1], [HCHO, O2], [CO, H2O]),
    Reaction("reaction_3", [2], [1, 1], [CH3OH], [DME, H2O]),
    Reaction("reaction_4", [2, 1], [1, 1], [CH3OH, HCHO], [DMM, H2O]),
    Reaction("reaction_5", [1, 1], [2, 1], [DME, O2], [HCHO, H2O]),
]


print(u_0 * rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0) * 2*r_part / (mu_gas_mix(T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2])))
print(u_0 * rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0) * 2*r_inner / (mu_gas_mix(T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2])))

def B_0(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T, d_t, d_p):
    return (
        G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T)
        * (
            (1 - porosity(d_t, d_p))
            / (
                rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0,
                        C_D0, C_E0, C_F0, C_G0, C_I0)
                * 2*r_part
                * (porosity(d_t, d_p) ** 3)
            )
        )
        * (
            (
                (
                    150
                    * (1 - porosity(d_t, d_p))
                    * mu_gas_mix(
                        T,
                        C_T,
                        [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I],
                        [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2],
                    )
                )
                / 2*r_part
            )
            + 1.75 * G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T)
        )
    )


def f(df, f, p, t):
    C_A, C_B, C_C, C_D, C_E, C_F, C_G, P, T_f, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs, T_s = f
    C_T = C_A + C_B + C_C + C_D + C_E + C_F + C_G + C_I0

    D_A_eff = CH3OH.D_eff(T_f, P_0, C_T, C_A, [C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2])
    D_B_eff = O2.D_eff(T_f, P_0, C_T, C_B, [C_A, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2])
    D_C_eff = HCHO.D_eff(T_f, P_0, C_T, C_C, [C_B, C_A, C_D, C_E, C_F, C_G, C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2])
    D_D_eff = H2O.D_eff(T_f, P_0, C_T, C_D, [C_B, C_C, C_A, C_E, C_F, C_G, C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2])
    D_E_eff = CO.D_eff(T_f, P_0, C_T, C_E, [C_A, C_B, C_C, C_D, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2])
    D_F_eff = DME.D_eff(T_f, P_0, C_T, C_F, [C_A, C_B, C_C, C_D, C_E, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2])
    D_G_eff = DMM.D_eff(T_f, P_0, C_T, C_G, [C_A, C_B, C_C, C_D, C_E, C_F, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2])

    n_1 = eta(
        theta(
            Reaction.k(A_HCHO, T_s, Ea_HCHO),
            D_A_eff,
            C_As + 1e-6,
            T_s,
            e1=-1,
        )
    )

    n_2 = eta(
        theta(
            Reaction.k(A_CO, T_s, Ea_CO),
            D_C_eff,
            1,
            T_s,
            e2=1,
        )
    )

    n_3 = eta(
        theta(
            Reaction.k(A_DMEf, T_s, Ea_DMEf),
            D_A_eff,
            1,
            T_s,
            e2=1,
        )
    )

    n_4 = eta(
        theta(
            Reaction.k(A_DMMf, T_s, Ea_DMMf),
            D_A_eff,
            C_As + 1e-6,
            T_s,
            e1=1,
            e2=2,
        )
    )

    n_5 = eta(
        theta(
            Reaction.k(A_DMEHCHO, T_s, Ea_DMEHCHO),
            D_F_eff,
            C_Fs + 1e-6,
            T_s,
            e1=-1,
        )
    )

    df[0] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            CH3OH.D_eff(T_f, P, C_T, C_A, [C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * (1/u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))
        * (C_As - C_A)
    )
    df[1] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            O2.D_eff(T_f, P, C_T, C_B, [C_A, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * (1/u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))
        * (C_Bs - C_B)
    )
    df[2] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            HCHO.D_eff(T_f, P, C_T, C_C, [C_B, C_A, C_D, C_E, C_F, C_G, C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * (1/u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))
        * (C_Cs - C_C)
    )
    df[3] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            H2O.D_eff(T_f, P, C_T, C_D, [C_B, C_C, C_A, C_E, C_F, C_G, C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * (1/u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))
        * (C_Ds - C_D)
    )
    df[4] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            CO.D_eff(T_f, P, C_T, C_E, [C_A, C_B, C_C, C_D, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * (1/u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))
        * (C_Es - C_E)
    )
    df[5] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            DME.D_eff(T_f, P, C_T, C_F, [C_A, C_B, C_C, C_D, C_E, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * (1/u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))
        * (C_Fs - C_F)
    )
    df[6] = (
        k_c(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2 * r_part,
            DMM.D_eff(T_f, P, C_T, C_G, [C_A, C_B, C_C, C_D, C_E, C_F, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * (1/u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))
        * (C_Gs - C_G)
    )
    
    df[7] = -B_0(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0, P, T_f, 2*r_inner, Dpe)*(T_f/T_0)*(P_0/P)*((C_T*u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0))/(C_T0*u_0))
    df[8] = (
        h(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2*r_part,
            kappa_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            Cp_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            )
        )
        * a_c(2*r_inner, 2*r_part)
        * (1/(u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0)*(CH3OH.Cp(T_f)*C_A + O2.Cp(T_f)*C_B + HCHO.Cp(T_f)*C_C + H2O.Cp(T_f)*C_D + CO.Cp(T_f)*C_E + DME.Cp(T_f)*C_F + DMM.Cp(T_f)*C_G + N2.Cp(T_f)*C_I0)))
        * (T_s - T_f)
    )

    df[9] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        CH3OH.D_eff(T_f, P, C_T, C_A, [C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_A - C_As) - (
        rho_cat * ((1-por)**2/por) * (n_1*r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + n_2*2*r3.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + n_4*2*r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[10] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        O2.D_eff(T_f, P, C_T, C_B, [C_A, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_B - C_Bs) - (
        0.5 * rho_cat * ((1-por)**2/por) * (n_1*r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + n_2*r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + n_5*r5.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[11] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        HCHO.D_eff(T_f, P, C_T, C_C, [C_B, C_A, C_D, C_E, C_F, C_G, C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_C - C_Cs) + (
        rho_cat * ((1-por)**2/por) * (n_1*r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + n_5*r5.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - n_2*r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - n_4*r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[12] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        H2O.D_eff(T_f, P, C_T, C_D, [C_B, C_C, C_A, C_E, C_F, C_G, C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_D - C_Ds) + (
        rho_cat * ((1-por)**2/por) * (n_1*r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + n_2*r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + n_3*r3.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + n_4*r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + 0.5*r5.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[13] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        CO.D_eff(T_f, P, C_T, C_E, [C_A, C_B, C_C, C_D, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_E - C_Es) + (
        rho_cat * ((1-por)**2/por) * n_2*r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)
    )
    df[14] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        DME.D_eff(T_f, P, C_T, C_F, [C_A, C_B, C_C, C_D, C_E, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_F - C_Fs) + (
        rho_cat * ((1-por)**2/por) * (n_3*r3.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - n_2*0.5*r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )

    df[15] = k_c(
        rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        mu_gas_mix(
            T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
        2 * r_part,
        DMM.D_eff(T_f, P, C_T, C_G, [C_A, C_B, C_C, C_D, C_E, C_F, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C_G - C_Gs) + (
        rho_cat * ((1-por)**2/por) * n_4*r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)
    )
    df[16] = (
        ((por/(1-por))*h(
            rho_mix(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            mu_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(T_f, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0),
            2*r_part,
            kappa_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            Cp_gas_mix(
                T_f, C_T, [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            )
        )
        * a_c(2*r_inner, 2*r_part) * (T_f - T_s)) + (((1-por)**2/por)*rho_cat*((-n_1*r1.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r1.H_rxn(T_s)) + (-n_2*r2.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r2.H_rxn(T_s)) + (-n_3*r3.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r3.H_rxn(T_s)) + (-n_4*r4.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r4.H_rxn(T_s)) + (-n_5*r5.r(T_s, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)*r5.H_rxn(T_s))))
    )
    
    return df

def condition(u, t, integrator):
    return u[14]
    
def affect_b(integrator):
    de.terminate_b(integrator) 

cb = de.ContinuousCallback(condition, affect_b)

f0 = [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, P_0, T_0, C_A0, C_Bs0, C_Cs0, C_Ds0, C_Es0, C_Fs0, C_Gs0, Ts_0]

M = mass_mat(17, 9)
z_span = (0, reactor_len(w_cat))

my_func = de.ODEFunction(f, mass_matrix=M)
prob_mm = de.ODEProblem(my_func, f0, z_span)
sol = de.solve(
    prob_mm,
    de.Rodas5P(autodiff=False),
    saveat=0.01,
    reltol=1e-6,
    abstol=1e-6,
    callback=cb
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
u_tot = u_A + u_B + u_C + u_D + u_E + u_F + u_G + C_I0

saf = [u_C[i]/(u_C[i] + u_E[i] + u_F[i] + u_G[i])*100 for i in range(1, len(u_A))]
yaf = [u_C[i]/(C_A0-u_A[i])*100 for i in range(1, len(u_A))]

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 9))

ax1.plot(z, u_A, z, u_C, z, u_D, z, u_E, z, u_F, z, u_G, linewidth=0.8)
ax1.set_title("Concentration in gas phase vs reactor length")
ax1.tick_params(axis="both",direction="in")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlabel("Reactor length, z (m)")
ax1.set_ylabel("Concentration, C (mol/m³)")
ax1.legend([r"$C_{CH_3OH}$", r"$C_{HCHO}$", r"$C_{H_2O}$", r"$C_{CO}$", r"$C_{DME}$", r"$C_{DMM}$"])
ax1.grid(color='0.8')

ax2.plot(z, u_As, z, u_Cs, z, u_Ds, z, u_Es, z, u_Fs, z, u_Gs, linewidth=0.8)
ax2.set_title("Concentration on catalyst surface vs reactor length")
ax2.tick_params(axis="both",direction="in")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlabel("Reactor length, z (m)")
ax2.set_ylabel(r"$C_{cat}$ (mol/m³)")
ax2.legend([r"$C_{CH_3OH}$", r"$C_{HCHO}$", r"$C_{H_2O}$", r"$C_{CO}$", r"$C_{DME}$", r"$C_{DMM}$"])
ax2.grid(color='0.8')

ax3.plot(z, (np.array(u_P))*1e2/P_0, linewidth=0.8)
ax3.tick_params(axis="both",direction="in")
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_xlabel("Reactor length, z (m)")
ax3.set_ylabel(r"$\frac{P}{P_0}x100$ (%)")
ax3.grid(color='0.8')

ax4.plot(z, u_T, z, u_Ts, linewidth=0.8)
ax4.tick_params(axis="both",direction="in")
ax4.spines[["top", "right"]].set_visible(False)
ax4.set_xlabel("Reactor length, z (m)")
ax4.set_ylabel("Temperature (K)")
ax4.legend([r"$T_{g}$", r"$T_{s}$"])
ax4.grid(color='0.8')

ax5.plot(z[1:], saf, linewidth=0.8)
ax5.tick_params(axis="both",direction="in")
ax5.spines[["top", "right"]].set_visible(False)
ax5.set_xlabel("Reactor length, z (m)")
ax5.set_ylabel("Selectivity (%)")
ax5.grid(color='0.8')

ax6.plot(z[1:], yaf, linewidth=0.8)
ax6.tick_params(axis="both",direction="in")
ax6.spines[["top", "right"]].set_visible(False)
ax6.set_xlabel("Reactor length, z (m)")
ax6.set_ylabel("Yield (%)")
ax6.grid(color='0.8')

plt.subplots_adjust(hspace=0.3)
plt.show()

print(f"\nReactor length = {reactor_len(w_cat)} m\n")

print(f"\nReactor entrance:\nC_CH₃OH = {sol(z[0])[0]} mol/m³\nC_O₂ = {sol(z[0])[1]} mol/m³\nC_HCHO = {sol(z[0])[2]} mol/m³\nC_H₂O = {sol(z[0])[3]} mol/m³\nC_CO = {sol(z[0])[4]} mol/m³\nC_DME = {sol(z[0])[5]} mol/m³\nC_DMM = {sol(z[0])[6]} mol/m³\nCs_CH₃OH = {sol(z[0])[9]} mol/m³\nCs_O₂ = {sol(z[0])[10]} mol/m³\nCs_HCHO = {sol(z[0])[11]} mol/m³\nCs_H₂O = {sol(z[0])[12]} mol/m³\nCs_CO = {sol(z[0])[13]} mol/m³\nCs_DME = {sol(z[0])[14]} mol/m³\nCs_DMM = {sol(z[0])[15]} mol/m³\nP = {sol(z[0])[7]} Pa\nT_f = {sol(z[0])[8]} K\nT_s = {sol(z[0])[16]} K\n")

print(f"\nReactor exit:\nC_CH₃OH = {sol(z[-1])[0]} mol/m³\nC_O₂ = {sol(z[-1])[1]} mol/m³\nC_HCHO = {sol(z[-1])[2]} mol/m³\nC_H₂O = {sol(z[-1])[3]} mol/m³\nC_CO = {sol(z[-1])[4]} mol/m³\nC_DME = {sol(z[-1])[5]} mol/m³\nC_DMM = {sol(z[-1])[6]} mol/m³\nCs_CH₃OH = {sol(z[-1])[9]} mol/m³\nCs_O₂ = {sol(z[-1])[10]} mol/m³\nCs_HCHO = {sol(z[-1])[11]} mol/m³\nCs_H₂O = {sol(z[-1])[12]} mol/m³\nCs_CO = {sol(z[-1])[13]} mol/m³\nCs_DME = {sol(z[-1])[14]} mol/m³\nCs_DMM = {sol(z[-1])[15]} mol/m³\nP = {sol(z[-1])[7]} Pa\nT_f = {sol(z[-1])[8]} K\nT_s = {sol(z[-1])[16]} K\n")

print(f"Conversion of Methanol: {round((u_A[0]-u_A[-1])*1e2/u_A[0], 4)}%")

m_in = q_dot(u_T[0], u_P[0], u_A[0]+u_B[0]+u_C[0]+u_D[0]+u_E[0]+u_F[0]+u_G[0]+C_I0, u_A[0], u_B[0], u_C[0], u_D[0], u_E[0], u_F[0], u_G[0], C_I0)*rho_mix(u_T[0], u_P[0], u_A[0]+u_B[0]+u_C[0]+u_D[0]+u_E[0]+u_F[0]+u_G[0]+C_I0, u_A[0], u_B[0], u_C[0], u_D[0], u_E[0], u_F[0], u_G[0], C_I0)

m_out = q_dot(u_T[-1], u_P[-1], u_A[-1]+u_B[-1]+u_C[-1]+u_D[-1]+u_E[-1]+u_F[-1]+u_G[-1]+C_I0, u_A[-1], u_B[-1], u_C[-1], u_D[-1], u_E[-1], u_F[-1], u_G[-1], C_I0)*rho_mix(u_T[-1], u_P[-1], u_A[-1]+u_B[-1]+u_C[-1]+u_D[-1]+u_E[-1]+u_F[-1]+u_G[-1]+C_I0, u_A[-1], u_B[-1], u_C[-1], u_D[-1], u_E[-1], u_F[-1], u_G[-1], C_I0)

print(m_in, m_out)
print("m_in - m_out =", m_in-m_out)

Y_theo = ((C_A0 * q_dot(u_T[0], u_P[0], u_tot[0], u_A[0], u_B[0], u_C[0], u_D[0], u_E[0], u_F[0], u_G[0], C_I0)) - (u_A[-1] * q_dot(u_T[0], u_P[0], u_tot[0], u_A[0], u_B[0], u_C[0], u_D[0], u_E[0], u_F[0], u_G[0], C_I0)))
print(f"\nTheoretical yield = {Y_theo}\n")

Y_real = u_C[-1] * q_dot(u_T[0], u_P[0], u_tot[0], u_A[0], u_B[0], u_C[0], u_D[0], u_E[0], u_F[0], u_G[0], C_I0)
print(f"\nReal yield = {Y_real}\n")

print("Yield = ", Y_real/Y_theo * 100)

selectivity = u_C[-1]/(u_C[-1] + u_E[-1] + u_F[-1] + u_G[-1])

print("Selectivity = ", selectivity)

print(u_P[-1]/P_0)


# np.savetxt("test.txt", (m_in, m_out))
