from scipy.integrate import quad
import numpy as np

R = 8.314


class Molecule:

    def __init__(self, name: str, M_w: float, H_f: float, params_vis: list, params_cp: list):
        self.name = name
        self.M_w = M_w
        self.H_f = H_f
        self.params_vis = params_vis
        self.params_cp = params_cp

    def mu(self, T: float, P: float) -> float:
        if self.name == "DMM":
            X = 2.12574 + 2063.71/(T*1.8) + 0.00119260*self.M_w
            K = (16.7175+0.0419188*self.M_w)*((T*1.8)**1.40256)/(212.209 + 18.1349*self.M_w + T*1.8)
            Y = 2.447 + 0.0392851*X
            return 1e-7*K*np.exp(X * self.rho(P*101325*1e-6, T)**Y)  # from article by O. Jeje and L. Mattar
        else:
            return self.params_vis[0]*(T**self.params_vis[1])/(1 + (self.params_vis[2]/T))  # from perry 2-267

    def rho(self, P: float, T: float) -> float:
        return self.M_w*(P/(R*T))

    def Cp(self, T: float) -> float:
        if self.name == "DMM":
            return 51.161 + 0.16244*T + 8.26e-5*(T**2) + (-8.51e-8*(T**3))  # C_p for DMM method of Joback parameters found in The properties of gases and liquids 
        else:
            return (self.params_cp[0] + (self.params_cp[1]*(self.params_cp[2]/(T*np.sinh(self.params_cp[2]/T)))**2) + (self.params_cp[3]*(self.params_cp[4]/(T*np.cosh(self.params_cp[4]/T)))**2))*1e-3  # from perry 2-149


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

    def r(self, T, P, F_A, F_B, F_C, F_D, F_E, F_F, F_G, F_T):
        if self.name == "reaction_1":
            return (
                Reaction.k(Reaction.A_HCHO, T, Reaction.Ea_HCHO)
                * (
                    (Reaction.k(Reaction.A_CH3OH, T, Reaction.Ea_CH3OH) * (P * F_A / F_T))
                    / (
                        1
                        + Reaction.k(Reaction.A_CH3OH, T, Reaction.Ea_CH3OH) * (P * F_A / F_T)
                        + Reaction.k(Reaction.A_H2O, T, Reaction.Ea_H2O) * (P * F_D / F_T)
                    )
                )
                * (
                    (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (P * F_B / F_T) ** 0.5)
                    / (1 + (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (P * F_B / F_T) ** 0.5))
                )
            )
        elif self.name == "reaction_2":
            return (
                Reaction.k(Reaction.A_CO, T, Reaction.Ea_CO)
                * (
                    (P * F_C / F_T)
                    / (
                        1
                        + Reaction.k(Reaction.A_CH3OH, T, Reaction.Ea_CH3OH) * (P * F_A / F_T)
                        + Reaction.k(Reaction.A_H2O, T, Reaction.Ea_H2O) * (P * F_D / F_T)
                    )
                )
                * (
                    (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (P * F_B / F_T) ** 0.5)
                    / (1 + (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (P * F_B / F_T) ** 0.5))
                )
            )
        elif self.name == "reaction_3":
            return Reaction.k(Reaction.A_DMEf, T, Reaction.Ea_DMEf) * (P * F_A / F_T) - (Reaction.k(Reaction.A_DMEf, T, Reaction.Ea_DMEf) / Reaction.K_eq_DME(T)) * (P * F_F * F_D / (F_A * F_T))
        elif self.name == "reaction_4":
            return Reaction.k(Reaction.A_DMMf, T, Reaction.Ea_DMMf) * (P**2 * F_A * F_C/F_T**2) - (Reaction.k(Reaction.A_DMMf, T, Reaction.Ea_DMMf)/Reaction.K_eq_DMM(T))*(P*F_D*F_G/(F_A*F_T))
        elif self.name == "reaction_5":
            return Reaction.k(Reaction.A_DMEHCHO, T, Reaction.Ea_DMEHCHO)*(Reaction.k(Reaction.A_DME, T, Reaction.Ea_DME)*(P*F_F/F_T)/(1 + Reaction.k(Reaction.A_DME, T, Reaction.Ea_DME)*(P*F_F/F_T)))*((Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (P * F_B / F_T) ** 0.5)/(1 + (Reaction.k(Reaction.A_O2, T, Reaction.Ea_O2) * (P * F_B / F_T) ** 0.5)))
        else:
            return "Unknown reaction"
