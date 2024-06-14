from diffeqpy import de
import matplotlib.pyplot as plt
from functions import *
from classes import *

CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2 = [
    Molecule("Methanol", Mw_Me, H_f_Me, Tb_Me, Vb_Me, Param_Mu_Me, Param_Cp_Me, Param_kappa_Me),
    Molecule("Oxygen", Mw_O2, H_f_O2, Tb_O2, Vb_O2, Param_Mu_O2, Param_Cp_O2, Param_kappa_O2),
    Molecule("Formaldehyde", Mw_HCHO, H_f_HCHO, Tb_HCHO, Vb_HCHO, Param_Mu_HCHO, Param_Cp_HCHO, Param_kappa_HCHO),
    Molecule("Water", Mw_H2O, H_f_H2O, Tb_H2O, Vb_H2O, Param_Mu_H2O, Param_Cp_H2O, Param_kappa_H2O),
    Molecule("Carbon Monoxide", Mw_CO, H_f_CO, Tb_CO, Vb_CO, Param_Mu_CO, Param_Cp_CO, Param_kappa_CO),
    Molecule("DME", Mw_DME, H_f_DME, Tb_DME, Vb_DME, Param_Mu_DME, Param_Cp_DME, Param_kappa_DME),
    Molecule("DMM", Mw_DMM, H_f_DMM, Tb_DMM, Vb_DMM, Param_Mu_DMM, [0, 0, 0, 0], Param_kappa_DMM),
    Molecule("Nitrogen", Mw_N2, H_f_N2, Tb_N2, Vb_N2, Param_Mu_N2, Param_Cp_N2, Param_kappa_N2),
]

print(H2O.Cp(400))
r1, r2, r3, r4, r5 = [
    Reaction("reaction_1", [1, 0.5], [1, 1], [CH3OH, O2], [HCHO, H2O]),
    Reaction("reaction_2", [1, 0.5], [1, 1], [HCHO, O2], [CO, H2O]),
    Reaction("reaction_3", [2], [1, 1], [CH3OH], [DME, H2O]),
    Reaction("reaction_4", [2, 1], [1, 1], [CH3OH, HCHO], [DMM, H2O]),
    Reaction("reaction_5", [1, 1], [2, 1], [DME, O2], [HCHO, H2O]),
]


def B_0(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T, d_t, d_p):
    return (
        G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T)
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
                    * mu_gas_mix(
                        T,
                        C_T,
                        [C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I],
                        [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2],
                    )
                )
                / d_p
            )
            + 1.75 * G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T)
        )
    )


def dCdw(C, p, t):
    C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T = C
    C_T = C_A + C_B + C_C + C_D + C_E + C_F + C_G + C_I

    print(T)

    return [
        -(0.2*r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 2*0.95*r3.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 2*0.8*r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G))/q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        -0.5*(0.2*r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.9*r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.3*r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G))/q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        ((0.2*r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.3*r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)) - (0.9*r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.8*r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)))/q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        (0.2*r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.9*r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.95*r3.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.8*r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.5*0.3*r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G))/q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        0.9*r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)/q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        (0.95*r3.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) - 0.5*0.3*r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G))/q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        0.8*r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)/q_dot(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        0,
        0, # -(B_0(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T, 2*r_inner, 2*r_part)/(A_c(r_inner)*(1-porosity(2*r_inner, 2*r_part))*rho_cat))*(T/T_0)*(P_0/P)*((C_T*u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I))/(C_T0*u_0)),
        ((-0.2*r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*0.2*r1.H_rxn(T))+(-0.9*r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*0.9*r2.H_rxn(T))+(-0.95*r3.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*0.95*r3.H_rxn(T))+(-0.8*r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*0.8*r4.H_rxn(T))+(-0.3*r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*0.3*r5.H_rxn(T)))/(F(T, P, C_A, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*CH3OH.Cp(T) + F(T, P, C_B, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*O2.Cp(T) + F(T, P, C_C, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*HCHO.Cp(T) + F(T, P, C_D, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*H2O.Cp(T) + F(T, P, C_E, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*CO.Cp(T) + F(T, P, C_F, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*DME.Cp(T) + F(T, P, C_G, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*DMM.Cp(T) + F(T, P, C_I, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*N2.Cp(T)),
    ]


def condition(out, u, t, integrator):
    out[0] = u[0]
    out[1] = u[1]
    out[2] = u[2]
    out[3] = u[3]
    out[4] = u[4]
    out[5] = u[5]
    out[6] = u[6]
    out[8] = u[8] - 1000
    out[9] = u[9] - 700
    return out


def affect_b(integrator, idx):
    if idx == 1:
        de.terminate_b(integrator)
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
    elif idx == 9:
        de.terminate_b(integrator)
    elif idx == 10:
        de.terminate_b(integrator)



cb = de.VectorContinuousCallback(condition, affect_b, 10)

w_span = (0, w_cat)
C_0 = [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0, P_0, T_0]

prob = de.ODEProblem(dCdw, C_0, w_span)
sol = de.solve(prob, de.Tsit5(), callback = cb, saveat=0.001, reltol=1e-8, abstol=1e-8)

w = sol.t
u_vals = np.array([sol(i) for i in w]).T

Y_A = u_vals[0]
Y_B = u_vals[1]
Y_C = u_vals[2]
Y_D = u_vals[3]
Y_E = u_vals[4]
Y_F = u_vals[5]
Y_G = u_vals[6]

Y_P = u_vals[8]
Y_T = u_vals[9]


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

ax1.plot(w, Y_A, w, Y_B, w, Y_C, w, Y_D, w, Y_E, w, Y_F, w, Y_G, linewidth=0.9)
ax1.tick_params(axis="both",direction="in")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlabel("catalyst weight, W [kg]")
ax1.set_ylabel("Concentration, C [mol/m³]")
ax1.legend([r"$C_{CH_3OH}$", r"$C_{O_2}$", r"$C_{HCHO}$", r"$C_{H_2O}$", r"$C_{CO}$", r"$C_{DME}$", r"$C_{DMM}$"], loc="center left")
ax1.grid(color='0.8')

ax2.plot(w, (P_0-np.array(Y_P))*1e2/P_0, linewidth=0.9)
ax2.tick_params(axis="both",direction="in")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlabel("catalyst weight, W [kg]")
ax2.set_ylabel("Pressure drop [%]")
ax2.grid(color='0.8')

ax3.plot(w, Y_T, linewidth=0.9)
ax3.tick_params(axis="both",direction="in")
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_xlabel("catalyst weight, W [kg]")
ax3.set_ylabel("Temperature [K]")
ax3.grid(color='0.8')

plt.show()

print(f"\nReactor entrance:\nC_CH₃OH = {sol(w[0])[0]} mol/m³\nC_O₂ = {sol(w[0])[1]} mol/m³\nC_HCHO = {sol(w[0])[2]} mol/m³\nC_H₂O = {sol(w[0])[3]} mol/m³\nC_CO = {sol(w[0])[4]} mol/m³\nC_DME = {sol(w[0])[5]} mol/m³\nC_DMM = {sol(w[0])[6]} mol/m³\nP = {sol(w[0])[8]} Pa\nT = {sol(w[0])[9]} K\n")

print(f"\nReactor exit:\nC_CH₃OH = {sol(w[-1])[0]} mol/m³\nC_O₂ = {sol(w[-1])[1]} mol/m³\nC_HCHO = {sol(w[-1])[2]} mol/m³\nC_H₂O = {sol(w[-1])[3]} mol/m³\nC_CO = {sol(w[-1])[4]} mol/m³\nC_DME = {sol(w[-1])[5]} mol/m³\nC_DMM = {sol(w[-1])[6]} mol/m³\nP = {sol(w[-1])[8]} Pa\nT = {sol(w[-1])[9]} K\n")

print(f"Conversion of Methanol: {round((Y_A[0]-Y_A[-1])*1e2/Y_A[0], 4)}%")
