from scipy.integrate import quad
import numpy as np
from decimal import Decimal, getcontext

getcontext().prec = 200

R = 8.314


def phi_ij(n_i, n_j, M_wi, M_wj):  # Properties of gases and liquids
    return (1 + ((n_i/n_j)**0.5)*((M_wj/M_wi)**0.25))**2 / (8*(1 + (M_wi/M_wj)))**0.5


class Molecule:

    def __init__(self, name: str, M_w: float, H_f: float, T_boil: float, Vb: float, params_vis: list, params_cp: list, params_kappa: list):
        self.name = name
        self.M_w = M_w
        self.H_f = H_f
        self.T_boil = T_boil
        self.Vb = Vb
        self.params_vis = params_vis
        self.params_cp = params_cp
        self.params_kappa = params_kappa

    @staticmethod
    def mu_gas_mix(T, C_T, Conc, Molecules):  # Properties of gases and liquids
        comb = list(zip(Conc, Molecules))
        sum = 0

        for i, j in enumerate(comb):
            numerator = (j[0]/C_T)*j[1].mu(T)
            denom = 0
            for k, m in enumerate(comb):
                denom += (m[0]/C_T)*phi_ij(j[1].mu(T), m[1].mu(T), j[1].M_w, m[1].M_w)
            sum += numerator/denom

        return sum

    @staticmethod
    def kappa_gas_mix(T, C_T, Conc, Molecules):  # Motivation from Properties of gases and liquids
        comb = list(zip(Conc, Molecules))
        sum = 0
        for i in comb:
            sum += (i[0]/C_T) * i[1].kappa(T)

        return sum

    @staticmethod
    def Cp_gas_mix(T, C_T, Conc, Molecules):  # Motivation from Properties of gases and liquids
        comb = list(zip(Conc, Molecules))
        sum = 0
        for i in comb:
            sum += (i[0]/C_T) * i[1].Cp(T)

        return sum

    @staticmethod
    def char_len(V_a: float, V_b: float) -> float:  # properties of gases and liquids 11-3, 11-4
        return 0.59*((V_a**(1/3))+(V_b**(1/3)))

    @staticmethod
    def T_crit(T: float, T_a: float, T_b: float) -> float:  # properties of gases and liquids 11-3, 11-4
        return T/(1.15*(T_a*T_b)**0.5)

    @staticmethod
    def collision_integral(T: float, T_a: float, T_b: float) -> float:  # properties of gases and liquids 11-3, 11-4
        return (1.06036/(Molecule.T_crit(T, T_a, T_b)**0.15610)) + (0.19300/np.exp(0.47635*Molecule.T_crit(T, T_a, T_b))) + (1.03587/np.exp(1.52996*Molecule.T_crit(T, T_a, T_b))) + (1.76474/np.exp(3.89411*Molecule.T_crit(T, T_a, T_b)))

    @staticmethod
    def D_AB(T: float, P: float, T_a: float, T_b: float, V_a: float, V_b: float, M_wa: float, M_wb: float) -> float:  # properties of gases and liquids 11-3, 11-4
        return (3.03-(0.98/(2*((1/M_wa)+(1/M_wb))**-1)**0.5))*(T**(3/2))*1e-7/(((2*((1/M_wa)+(1/M_wb))**-1)**0.5)*(P/101325)*(Molecule.char_len(V_a, V_b)**2)*Molecule.collision_integral(T, T_a, T_b))

    def mu(self, T: float) -> float:
        if self.name == "DMM":
            return (self.params_vis[0] + self.params_vis[1]*T + self.params_vis[2]*T**2 + self.params_vis[3]*np.float_power(T, 3, dtype=np.longdouble))*1e-7   # from Transport properties of hydrocarbons
        else:
            return self.params_vis[0]*(T**self.params_vis[1])/(1 + (self.params_vis[2]/T))  # from perry 2-267

    def rho(self, P: float, T: float) -> float:
        return self.M_w*(abs(P)/(R*T))

    def Cp(self, T: float) -> float:
        if self.name == "DMM":
            if T > 1e40:
                print(f"Joback: {T}")
            return 51.161 + 0.16244*T + 8.26e-5*(T**2) + (-8.51e-8*(np.float_power(T, 3, dtype=np.longdouble)))  # C_p for DMM method of Joback parameters found in The properties of gases and liquids 
        else:
            if T > 1e40:
                print(T)
            return (self.params_cp[0] + (self.params_cp[1]*(self.params_cp[2]/(T*np.sinh(self.params_cp[2]/T, dtype=np.longdouble)))**2) + (self.params_cp[3]*(self.params_cp[4]/(T*np.cosh(self.params_cp[4]/T, dtype=np.longdouble)))**2))*1e-3  # from perry 2-149

    def kappa(self, T):  # perry 2-289 & Transport properties of hydrocarbons
        if self.name == "DMM":
            return self.params_kappa[0] + self.params_kappa[1]*T + self.params_kappa[2]*T**2 + self.params_kappa[3]*T**3
        else:
            return self.params_kappa[0]*T**self.params_kappa[1] / (1 + (self.params_kappa[2]/T) + (self.params_kappa[3]/T**2))

    def D_eff(self, T, P, C_T, C_main, Conc, Molecules):
        comb = list(zip(Conc, Molecules))
        denom = 0
        numerator = 1 - (C_main/C_T)
        for i in comb:
            denom += (i[0]/C_T)/Molecule.D_AB(T, P, self.T_boil, i[1].T_boil, self.Vb, i[1].Vb, self.M_w, i[1].M_w)

        return numerator / denom


class Reaction:

    A_CH3OH = 2.6e-4  # [atm^-1]
    A_O2 = 1.423e-5  # [atm^-0.5]
    A_H2O = 5.5e-7  # [atm^-1]
    A_HCHO = 1.5e7  # [mol/(kg_cat s)]
    A_CO = 3.5e2  # [mol/(kg_cat s atm)]
    A_DMEf = 1.9e5  # [mol/(kg_cat s atm)]
    A_DMMf = 4.26e-6  # [mol/(kg_cat s atm²)]
    A_DME = 5e-7  # [atm^-1]
    A_DMEHCHO = 6.13e5  # [mol/(kg_cat s)]

    Ea_CH3OH = -56780  # [J/mol]
    Ea_O2 = -60320  # [J/mol]
    Ea_H2O = -86450  # [J/mol]
    Ea_HCHO = 86000  # [J/mol]
    Ea_CO = 46000  # [J/mol]
    Ea_DMEf = 77000  # [J/mol]
    Ea_DMMf = 46500  # [J/mol]
    Ea_DME = -96720  # [J/mol]
    Ea_DMEHCHO = 98730  # [J/mol]


    def __init__(self, name: str, stoich_coeff_reac: list, stoich_coeff_prod: list, reactants: list, products: list):
        self.name = name
        self.stoich_coeff_reac = stoich_coeff_reac
        self.stoich_coeff_prod = stoich_coeff_prod
        self.reactants = reactants
        self.products = products

    # Equilibrium constants from Deshmuk
    @staticmethod
    def K_eq_DME(T):
        return np.exp(-2.2158 + (2606.8 / T))

    @staticmethod
    def K_eq_DMM(T):
        return np.exp(-20.416 + (9346.8 / T))

    # Arrhenius equation. Used for rate constant and adsorption constants (based on kinetics from Deshmuk).
    @staticmethod
    def k(A: float, T: float, Ea: float) -> float:
        return A * np.exp(-Ea / (R * T))

    def del_cp(self, T):
        reac = list(zip(self.stoich_coeff_reac, self.reactants))
        prod = list(zip(self.stoich_coeff_prod, self.products))

        reac_result = 0
        for i in reac:
            reac_result += (i[0]*i[1].Cp(T))

        prod_result = 0
        for i in prod:
            prod_result += (i[0]*i[1].Cp(T))

        return prod_result - reac_result

    def H_rxn(self, T, T_ref=298.15):
        reac = list(zip(self.stoich_coeff_reac, self.reactants))
        prod = list(zip(self.stoich_coeff_prod, self.products))

        reac_result = 0
        for i in reac:
            reac_result += (i[0]*i[1].H_f)

        prod_result = 0
        for i in prod:
            prod_result += (i[0]*i[1].H_f)

        return (prod_result - reac_result) + quad(self.del_cp, T_ref, T)[0]

    def r(self, T, C_A, C_B, C_C, C_D, C_E, C_F, C_G):
        if self.name == "reaction_1":
            return (
                Reaction.k(Reaction.A_HCHO, T, Reaction.Ea_HCHO)
                * (
                    (Reaction.k(Reaction.A_CH3OH, T, Reaction.Ea_CH3OH) * (C_A*R*T/101325))
                    / (
                        1
                        + Reaction.k(Reaction.A_CH3OH, T, Reaction.Ea_CH3OH) * (C_A*R*T/101325)
                        + Reaction.k(Reaction.A_H2O, T, Reaction.Ea_H2O) * (C_D*R*T/101325)
                    )
                )
                * (
                    (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (abs(C_B*R*T/101325)) ** 0.5)
                    / (1 + (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (abs(C_B*R*T/101325)) ** 0.5))
                )
            )
        elif self.name == "reaction_2":
            return (
                Reaction.k(Reaction.A_CO, T, Reaction.Ea_CO)
                * (
                    (C_C*R*T/101325)
                    / (
                        1
                        + Reaction.k(Reaction.A_CH3OH, T, Reaction.Ea_CH3OH) * (C_A*R*T/101325)
                        + Reaction.k(Reaction.A_H2O, T, Reaction.Ea_H2O) * (C_D*R*T/101325)
                    )
                )
                * (
                    (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (abs(C_B*R*T/101325)) ** 0.5)
                    / (1 + (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (abs(C_B*R*T/101325)) ** 0.5))
                )
            )
        elif self.name == "reaction_3":
            return Reaction.k(Reaction.A_DMEf, T, Reaction.Ea_DMEf) * (C_A*R*T/101325) - (Reaction.k(Reaction.A_DMEf, T, Reaction.Ea_DMEf) / Reaction.K_eq_DME(T)) * (C_F * C_D * R * T * (101325)**-1/ C_A)
        elif self.name == "reaction_4":
            return Reaction.k(Reaction.A_DMMf, T, Reaction.Ea_DMMf) * (C_A * C_C * (R*T/101325)**2) - (Reaction.k(Reaction.A_DMMf, T, Reaction.Ea_DMMf)/Reaction.K_eq_DMM(T))*(C_D*C_G*R*T*(101325)**-1/C_A)
        elif self.name == "reaction_5":
            return Reaction.k(Reaction.A_DMEHCHO, T, Reaction.Ea_DMEHCHO)*(Reaction.k(Reaction.A_DME, T, Reaction.Ea_DME)*(C_F*R*T/101325)/(1 + Reaction.k(Reaction.A_DME, T, Reaction.Ea_DME)*(C_F*R*T/101325)))*((Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (abs(C_B*R*T/101325)) ** 0.5)/(1 + (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (abs(C_B*R*T/101325)) ** 0.5)))
        else:
            return "Unknown reaction"
