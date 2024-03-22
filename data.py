import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

df1 = pl.read_csv("Air_density.csv")
df2 = pl.read_csv("Air_viscosity.csv")

temp_density, air_density = np.array(df1[0:, 1]), np.array(df1[0:, 2])

temp_viscosity, air_viscosity = np.array(df2[0:, 1]), np.array(df2[0:, 3])

rho_air = interp1d(temp_density, air_density, kind="cubic")

mu_air = interp1d(temp_viscosity, air_viscosity, kind="cubic")


def porosity(d_t, d_p):
    return 0.38 + 0.0073 * (1 + (((d_t / d_p) - 2) / (d_t / d_p)) ** 2)


temp_cp = np.array([300, 400, 500, 600, 800, 1000, 1500])
a1 = interp1d(temp_cp, [4.68, 0.864, -1.85, -4.61, -7.49, -8.53, -7.37], kind="cubic")
a2 = interp1d(temp_cp, [9.04, 12.6, 15.5, 17.5, 20.1, 21.6, 23.9], kind="cubic")
a3 = interp1d(temp_cp, [5.69, 7.37, 8.89, 10.5, 13.1, 15.2, 17.9], kind="cubic")
a4 = interp1d(temp_cp, [11.4, 13.9, 15.7, 17.5, 19.4, 20.4, 20.6], kind="cubic")


def C_p(T, n_C, n_H, n_O):
    return a1(T) + a2(T)*n_C + a3(T)*n_H + a4(T)*n_O



def C_p_HCHO(T):
    return C_p(T, 1, 2, 1)


def C_p_Me(T):
    return C_p(T, 1, 4, 1)


def C_p_H2O(T):
    return C_p(T, 0, 2, 1)


def C_p_O2(T):
    return C_p(T, 0, 0, 2)


def C_p_CO(T):
    return C_p(T, 1, 0, 1)


def C_p_DME(T):
    return C_p(T, 2, 6, 1)


def C_p_DMM(T):
    return C_p(T, 3, 8, 2)
