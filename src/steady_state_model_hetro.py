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


def B_0_new(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r, d_t, d_p):
    return (
        G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r)
        * (
            (1 - porosity(d_t, d_p))
            / (
                rho_mix(T_0, P_0, F_T0, F_A0, F_B0, F_C0,
                        F_D0, F_E0, F_F0, F_G0, F_I0)
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
                        F_T,
                        [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I],
                        [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2],
                    )
                )
                / d_p
            )
            + 1.75 * G(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I, P, T, r)
        )
    )


def f(df, f, p, t):
    F_A, F_B, F_C, F_D, F_E, F_F, F_G, P, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs = f
    F_T = F_A + F_B + F_C + F_D + F_E + F_F + F_G + F_I0

    df[0] = (
        k_c(
            rho_mix(T_0, P, F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0),
            Molecule.mu_gas_mix(
                T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(F_T, P, T_0, r_inner),
            2 * r_part,
            CH3OH.D_eff(T_0, P, F_T, F_A, [F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_As - C(F_A, F_T, P, T_0))
    )
    df[1] = (
        k_c(
            rho_mix(T_0, P, F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0),
            Molecule.mu_gas_mix(
                T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(F_T, P, T_0, r_inner),
            2 * r_part,
            O2.D_eff(T_0, P, F_T, F_B, [F_A, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Bs - C(F_B, F_T, P, T_0))
    )
    df[2] = (
        k_c(
            rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
            Molecule.mu_gas_mix(
                T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(F_T, P, T_0, r_inner),
            2 * r_part,
            HCHO.D_eff(T_0, P, F_T, F_C, [F_B, F_A, F_D, F_E, F_F, F_G, F_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Cs - C(F_C, F_T, P, T_0))
    )
    df[3] = (
        k_c(
            rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
            Molecule.mu_gas_mix(
                T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(F_T, P, T_0, r_inner),
            2 * r_part,
            H2O.D_eff(T_0, P, F_T, F_D, [F_B, F_C, F_A, F_E, F_F, F_G, F_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Ds - C(F_D, F_T, P, T_0))
    )
    df[4] = (
        k_c(
            rho_mix(T_0, P, F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0),
            Molecule.mu_gas_mix(
                T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(F_T, P, T_0, r_inner),
            2 * r_part,
            CO.D_eff(T_0, P, F_T, F_E, [F_B, F_C, F_D, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Es - C(F_E, F_T, P, T_0))
    )
    df[5] = (
        k_c(
            rho_mix(T_0, P, F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0),
            Molecule.mu_gas_mix(
                T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(F_T, P, T_0, r_inner),
            2 * r_part,
            DME.D_eff(T_0, P, F_T, F_F, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Fs - C(F_F, F_T, P, T_0))
    )
    df[6] = (
        k_c(
            rho_mix(T_0, P, F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0),
            Molecule.mu_gas_mix(
                T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
            ),
            u(F_T, P, T_0, r_inner),
            2 * r_part,
            DMM.D_eff(T_0, P, F_T, F_G, [F_B, F_C, F_D, F_E, F_F, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2]),
        )
        * a_c(2 * r_inner, 2 * r_part)
        * A_c(r_inner)
        * (C_Gs - C(F_G, F_T, P, T_0))
    )
    
    df[7] = -B_0_new(F_T, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0, P, T_0, r_inner, 2*r_inner, 2*r_part)*(T_0/T_0)*(P_0/P)*(F_T/F_T0)

    df[8] = k_c(
        rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
        Molecule.mu_gas_mix(
            T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(F_T, P, T_0, r_inner),
        2 * r_part,
        CH3OH.D_eff(T_0, P, F_T, F_A, [F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C(F_A, F_T, P, T_0) - C_As) - (
        rho_cat * (r1.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + 2*r3.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + 2*r4.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[9] = k_c(
        rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
        Molecule.mu_gas_mix(
            T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(F_T, P, T_0, r_inner),
        2 * r_part,
        O2.D_eff(T_0, P, F_T, F_B, [F_A, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C(F_B, F_T, P, T_0) - C_Bs) - (
        0.5 * rho_cat * (r1.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r2.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r5.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[10] = k_c(
        rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
        Molecule.mu_gas_mix(
            T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(F_T, P, T_0, r_inner),
        2 * r_part,
        HCHO.D_eff(T_0, P, F_T, F_C, [F_B, F_A, F_D, F_E, F_F, F_G, F_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C(F_C, F_T, P, T_0) - C_Cs) + (
        rho_cat * (r1.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r5.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - r2.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - r4.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[11] = k_c(
        rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
        Molecule.mu_gas_mix(
            T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(F_T, P, T_0, r_inner),
        2 * r_part,
        H2O.D_eff(T_0, P, F_T, F_D, [F_B, F_C, F_A, F_E, F_F, F_G, F_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C(F_D, F_T, P, T_0) - C_Ds) + (
        rho_cat * (r1.r(T_0, C_As, C_Bs, C_Cs, C_Ds, 0, 0, 0) + r2.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r3.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + r4.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) + 0.5*r5.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[12] = k_c(
        rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
        Molecule.mu_gas_mix(
            T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(F_T, P, T_0, r_inner),
        2 * r_part,
        CO.D_eff(T_0, P, F_T, F_E, [F_B, F_C, F_D, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C(F_E, F_T, P, T_0) - C_Es) + (
        rho_cat * r2.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)
    )
    df[13] = k_c(
        rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
        Molecule.mu_gas_mix(
            T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(F_T, P, T_0, r_inner),
        2 * r_part,
        DME.D_eff(T_0, P, F_T, F_F, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C(F_F, F_T, P, T_0) - C_Fs) + (
        rho_cat * (r3.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs) - 0.5*r2.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs))
    )
    df[14] = k_c(
        rho_mix(T_0, P, F_T, F_A, F_C, 0, 0, 0, F_D, F_B, F_I0),
        Molecule.mu_gas_mix(
            T_0, F_T, [F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
        ),
        u(F_T, P, T_0, r_inner),
        2 * r_part,
        DMM.D_eff(T_0, P, F_T, F_G, [F_B, F_C, F_D, F_E, F_F, F_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2]),
    ) * a_c(2 * r_inner, 2 * r_part) * (C(F_G, F_T, P, T_0) - C_Gs) + (
        rho_cat * r4.r(T_0, C_As, C_Bs, C_Cs, C_Ds, C_Es, C_Fs, C_Gs)
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
    out[8] = u[8]
    out[9] = u[9]
    out[10] = u[10]
    out[11] = u[11]
    out[12] = u[12]
    out[13] = u[13]
    out[14] = u[14]
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
        integrator.u[8] = 0
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


cb = de.VectorContinuousCallback(condition, affect_b, 15)

f0 = [F_A0, F_B0, F_C0, F_D0, F_E0, F_F0, F_G0, P_0, C_As0, C_Bs0, C_Cs0, C_Ds0, C_Es0, C_Fs0, C_Gs0]

M = mass_mat(15, 8)
z_span = (0, reactor_len(w_cat))


my_func = de.ODEFunction(f, mass_matrix=M)
prob_mm = de.ODEProblem(my_func, f0, z_span)
sol = de.solve(
    prob_mm,
    de.Rodas5P(autodiff=False),
    callback=cb,
    saveat=0.01,
    reltol=1e-8,
    abstol=1e-8,
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
u_As = u_vals[8]
u_Bs = u_vals[9]
u_Cs = u_vals[10]
u_Ds = u_vals[11]
u_Es = u_vals[12]
u_Fs = u_vals[13]
u_Gs = u_vals[14]


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

ax4.plot()
ax4.tick_params(axis="both",direction="in")
ax4.spines[["top", "right"]].set_visible(False)
ax4.set_xlabel("Reactor length, z [m]")
ax4.set_ylabel("Temperature [K]")
ax4.grid(color='0.8')


plt.subplots_adjust(hspace=0.3)
plt.show()


print("Reactor entrance:", sol(z[0]))
print("Reactor exit:", sol(z[-1]))
print(f"Conversion of Methanol: {(u_A[0]-u_A[-1])*1e2/u_A[0]}%")
