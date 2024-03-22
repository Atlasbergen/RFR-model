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
