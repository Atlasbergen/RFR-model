from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from functions import *
from classes import *

# print(F_A)
print("Reactor length: ", reactor_len(w_cat))
print("Reactor weight: ", reactor_weight(1))

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

r1, r2, r3, r4, r5 = [
    Reaction("reaction_1", [1, 0.5], [1, 1], [CH3OH, O2], [HCHO, H2O]),
    Reaction("reaction_2", [1, 0.5], [1, 1], [HCHO, O2], [CO, H2O]),
    Reaction("reaction_3", [2], [1, 1], [CH3OH], [DME, H2O]),
    Reaction("reaction_4", [2, 1], [1, 1], [CH3OH, HCHO], [DMM, H2O]),
    Reaction("reaction_5", [1, 1], [2, 1], [DME, O2], [HCHO, H2O]),
]

re = (u_0 * rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0) * 2*r_part / (mu_gas_mix(T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2])))
print(u_0 * rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0) * 2*r_part / (mu_gas_mix(T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2])))
print(u_0 * rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0) * 2*r_inner / (mu_gas_mix(T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2])))
print(1-porosity(2*r_inner, Dpe))
schmid = Sc(
    rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0), 
    mu_gas_mix(T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]),
    CH3OH.D_eff(T_0, P_0, C_T0, C_A0, [C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2]),
)
print("Sc =", schmid, por)

# pec = Pe(re, schmid)

# print("Peclet = ", pec)

epsilon = (rho_cat*(1 - por))

def B_0(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T, d_t, d_p):
    return (
        G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T)
        * (
            (1 - porosity(d_t, d_p))
            / (
                rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0,
                        C_D0, C_E0, C_F0, C_G0, C_I0)
                * (2*r_part)
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
                / (2*r_part)
            )
            + 1.75 * G(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T)
        )
    )


def dCdw(t, C):
    C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T = C
    C_T = C_A + C_B + C_C + C_D + C_E + C_F + C_G + C_I

    # print(T)

    if C_A <= 0:
        C_A = 0

    
    return [
        -epsilon * (r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 2*r3.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 2*r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G))/u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        -0.5 * epsilon * (r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G))/u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        epsilon * ((r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)) - (r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)))/u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        epsilon * (r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + r3.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) + 0.5*r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G))/u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        epsilon * r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)/u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        epsilon * (r3.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G) - 0.5*r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G))/u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        epsilon * r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)/u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I),
        0,
        -B_0(C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, P, T, 2*r_inner, Dpe)*(T/T_0)*(P_0/P)*((C_T*u(T, P, C_T, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I))/(C_T0*u_0)),
        epsilon*A_c(r_inner)*((-r1.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*r1.H_rxn(T))+(-r2.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*r2.H_rxn(T))+(-r3.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*r3.H_rxn(T))+(-r4.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*r4.H_rxn(T))+(-r5.r(T, C_A, C_B, C_C, C_D, C_E, C_F, C_G)*r5.H_rxn(T)))/(F(T, P, C_A, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*CH3OH.Cp(T) + F(T, P, C_B, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*O2.Cp(T) + F(T, P, C_C, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*HCHO.Cp(T) + F(T, P, C_D, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*H2O.Cp(T) + F(T, P, C_E, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*CO.Cp(T) + F(T, P, C_F, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*DME.Cp(T) + F(T, P, C_G, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*DMM.Cp(T) + F(T, P, C_I, C_A, C_B, C_C, C_D, C_E, C_F, C_G, C_I, C_T)*N2.Cp(T)),
   ]

w_span = (0, reactor_len(w_cat))
C_0 = [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0, P_0, T_0]

sol = solve_ivp(dCdw, w_span, C_0, method="RK45", rtol=1e-6, atol=1e-6)

w = sol.t  # reactor_len(sol.t)
u_vals = sol.y

z = sol.t

Y_A = u_vals[0]
Y_B = u_vals[1]
Y_C = u_vals[2]
Y_D = u_vals[3]
Y_E = u_vals[4]
Y_F = u_vals[5]
Y_G = u_vals[6]

Y_P = u_vals[8]
Y_T = u_vals[9]

r_1 = []
r_2 = []
r_3 = []
r_4 = []
r_5 = []

for i in range(len(sol.t)):
    r_1.append(r1.r(Y_T[i], Y_A[i], Y_B[i], Y_C[i], Y_D[i], Y_E[i], Y_F[i], Y_G[i]))
   

for i in range(len(sol.t)):
    r_2.append(r2.r(Y_T[i], Y_A[i], Y_B[i], Y_C[i], Y_D[i], Y_E[i], Y_F[i], Y_G[i]))


for i in range(len(sol.t)):
    r_3.append((r1.r(Y_T[i], Y_A[i], Y_B[i], Y_C[i], Y_D[i], Y_E[i], Y_F[i], Y_G[i]) + r5.r(Y_T[i], Y_A[i], Y_B[i], Y_C[i], Y_D[i], Y_E[i], Y_F[i], Y_G[i])) - (r2.r(Y_T[i], Y_A[i], Y_B[i], Y_C[i], Y_D[i], Y_E[i], Y_F[i], Y_G[i]) + r4.r(Y_T[i], Y_A[i], Y_B[i], Y_C[i], Y_D[i], Y_E[i], Y_F[i], Y_G[i])))


for i in range(len(sol.t)):
    r_4.append(r4.r(Y_T[i], Y_A[i], Y_B[i], Y_C[i], Y_D[i], Y_E[i], Y_F[i], Y_G[i]))


for i in range(len(sol.t)):
    r_5.append(r5.r(Y_T[i], Y_A[i], Y_B[i], Y_C[i], Y_D[i], Y_E[i], Y_F[i], Y_G[i]))


saf = [Y_C[i]/(Y_C[i] + Y_E[i] + Y_F[i] + Y_G[i])*100 for i in range(1, len(Y_A))]
yaf = [Y_C[i]/(C_A0-Y_A[i])*100 for i in range(1, len(Y_A))]

fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(20, 16))

ax1.plot(z, Y_A, z, Y_C, z, Y_D, z, Y_E, z, Y_F, z, Y_G, linewidth=0.8)
ax1.tick_params(axis="both",direction="in")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlabel("Reactor length, z (m)")
ax1.set_ylabel("Concentration, C (mol/m³)")
ax1.legend([r"$C_{CH_3OH}$", r"$C_{HCHO}$", r"$C_{H_2O}$", r"$C_{CO}$", r"$C_{DME}$", r"$C_{DMM}$"], loc="center left") #, bbox_to_anchor=(1, 0.0), ncol=1)
ax1.grid(color='0.8')

ax2.plot(z, (np.array(Y_P))*100/P_0, linewidth=0.8)
ax2.tick_params(axis="both",direction="in")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlabel("Reactor length, z (m)")
ax2.set_ylabel(r"$\frac{P}{P_0}x100$ (%)")
ax2.grid(color='0.8')

ax3.plot(z, Y_T, linewidth=0.8)
ax3.tick_params(axis="both",direction="in")
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_xlabel("Reactor length, z (m)")
ax3.set_ylabel("Temperature (K)")
ax3.grid(color='0.8')

ax4.plot(z, np.array(r_1)*epsilon, linewidth=0.8) #, z, r_2, z, r_3, z, r_4, z, r_5, linewidth=0.9)
ax4.tick_params(axis="both",direction="in")
ax4.spines[["top", "right"]].set_visible(False)
ax4.set_xlabel("Reactor length, z (m)")
ax4.set_ylabel("Reaction rate (mol m⁻³ s⁻¹)")
ax4.grid(color='0.8')
ax4.legend(["reaction rate, reaction [A]"], loc="upper right")

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
plt.show()

print(f"\nReactor entrance:\nC_CH₃OH = {Y_A[0]} mol/m³\nC_O₂ = {Y_B[0]} mol/m³\nC_HCHO = {Y_C[0]} mol/m³\nC_H₂O = {Y_D[0]} mol/m³\nC_CO = {Y_E[0]} mol/m³\nC_DME = {Y_F[0]} mol/m³\nC_DMM = {Y_G[0]} mol/m³\nP = {Y_P[0]} Pa\nT = {Y_T[0]} K\n")

print(f"\nReactor exit:\nC_CH₃OH = {Y_A[-1]} mol/m³\nC_O₂ = {Y_B[-1]} mol/m³\nC_HCHO = {Y_C[-1]} mol/m³\nC_H₂O = {Y_D[-1]} mol/m³\nC_CO = {Y_E[-1]} mol/m³\nC_DME = {Y_F[-1]} mol/m³\nC_DMM = {Y_G[-1]} mol/m³\nP = {Y_P[-1]} Pa\nT = {Y_T[-1]} K\n")

print(f"Conversion of Methanol: {round((Y_A[0]-Y_A[-1])*1e2/Y_A[0], 4)}%")

m_in = q_dot(Y_T[0], Y_P[0], Y_A[0]+Y_B[0]+Y_C[0]+Y_D[0]+Y_E[0]+Y_F[0]+Y_G[0]+C_I0, Y_A[0], Y_B[0], Y_C[0], Y_D[0], Y_E[0], Y_F[0], Y_G[0], C_I0)*rho_mix(Y_T[0], Y_P[0], Y_A[0]+Y_B[0]+Y_C[0]+Y_D[0]+Y_E[0]+Y_F[0]+Y_G[0]+C_I0, Y_A[0], Y_B[0], Y_C[0], Y_D[0], Y_E[0], Y_F[0], Y_G[0], C_I0)

m_out = q_dot(Y_T[-1], Y_P[-1], Y_A[-1]+Y_B[-1]+Y_C[-1]+Y_D[-1]+Y_E[-1]+Y_F[-1]+Y_G[-1]+C_I0, Y_A[-1], Y_B[-1], Y_C[-1], Y_D[-1], Y_E[-1], Y_F[-1], Y_G[-1], C_I0)*rho_mix(Y_T[-1], Y_P[-1], Y_A[-1]+Y_B[-1]+Y_C[-1]+Y_D[-1]+Y_E[-1]+Y_F[-1]+Y_G[-1]+C_I0, Y_A[-1], Y_B[-1], Y_C[-1], Y_D[-1], Y_E[-1], Y_F[-1], Y_G[-1], C_I0)

# print("m_in - m_out =", m_in-m_out)

print("Selectivity = ", Y_C[-1]/(Y_C[-1] + Y_E[-1] + Y_F[-1] + Y_G[-1]))
print("Yield = ", Y_C[-1]/(C_A0 - Y_A[-1]))

# u_end = (u(Y_T[-1], Y_P[-1], Y_A[-1]+Y_B[-1]+Y_C[-1]+Y_D[-1]+Y_E[-1]+Y_F[-1]+Y_G[-1]+C_I0, Y_A[-1], Y_B[-1], Y_C[-1], Y_D[-1], Y_E[-1], Y_F[-1], Y_G[-1], C_I0))
# u_star = (u(Y_T[0], Y_P[0], Y_A[0]+Y_B[0]+Y_C[0]+Y_D[0]+Y_E[0]+Y_F[0]+Y_G[0]+C_I0, Y_A[0], Y_B[0], Y_C[0], Y_D[0], Y_E[0], Y_F[0], Y_G[0], C_I0))
# print(u_end/u_star)


# p_end = (rho_mix(Y_T[-1], Y_P[-1], Y_A[-1]+Y_B[-1]+Y_C[-1]+Y_D[-1]+Y_E[-1]+Y_F[-1]+Y_G[-1]+C_I0, Y_A[-1], Y_B[-1], Y_C[-1], Y_D[-1], Y_E[-1], Y_F[-1], Y_G[-1], C_I0))
# p_star = (rho_mix(Y_T[0], Y_P[0], Y_A[0]+Y_B[0]+Y_C[0]+Y_D[0]+Y_E[0]+Y_F[0]+Y_G[0]+C_I0, Y_A[0], Y_B[0], Y_C[0], Y_D[0], Y_E[0], Y_F[0], Y_G[0], C_I0))
# print((p_end)/p_star)
# print((r1.r(T_0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0)*2*r_part/(C_A0*u_0)), (u_0*2*r_part/1e-5))

print(h(
    rho_mix(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0),
    mu_gas_mix(
        T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
    ),
    u(T_0, P_0, C_T0, C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0),
    2*r_part,
    kappa_gas_mix(
        T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
    ),
    Cp_gas_mix(
        T_0, C_T0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, DMM, N2]
    )
)*r_part*2/thermcond_cat(Ts_0))


D_A_eff = CH3OH.D_eff(T_0, P_0, C_T0, C_A0, [C_B0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [O2, HCHO, H2O, CO, DME, DMM, N2])
D_B_eff = O2.D_eff(T_0, P_0, C_T0, C_B0, [C_A0, C_C0, C_D0, C_E0, C_F0, C_G0, C_I0], [CH3OH, HCHO, H2O, CO, DME, DMM, N2])
D_C_eff = HCHO.D_eff(T_0, P_0, C_T0, C_C0, [C_B0, C_A0, C_D0, C_E0, C_F0, C_G0, C_I0], [O2, CH3OH, H2O, CO, DME, DMM, N2])
D_D_eff = H2O.D_eff(T_0, P_0, C_T0, C_D0, [C_B0, C_C0, C_A0, C_E0, C_F0, C_G0, C_I0], [O2, HCHO, CH3OH, CO, DME, DMM, N2])
D_E_eff = CO.D_eff(T_0, P_0, C_T0, C_E0, [C_A0, C_B0, C_C0, C_D0, C_F0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, DME, DMM, N2])
D_F_eff = DME.D_eff(T_0, P_0, C_T0, C_F0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_G0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DMM, N2])
D_G_eff = DMM.D_eff(T_0, P_0, C_T0, C_G0, [C_A0, C_B0, C_C0, C_D0, C_E0, C_F0, C_I0], [CH3OH, O2, HCHO, H2O, CO, DME, N2])

print(u_0*2*r_part/D_A_eff)

# print(Y_P[-1]/P_0)
