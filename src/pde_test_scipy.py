import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m = 60
snaps = 600

L = 2
dx = L / m
dt = 0.1

v0 = 0.1 
k = 0.2


def deriv(t, u):
    dudt = np.zeros_like(u)
    for i in range(1, m):
        dudt[i] = -v0 * ((u[i] - u[i-1]) / (dx)) - k*u[i]
        dudt[i + m] = -v0 * ((u[i+m] - u[i-1+m]) / (dx)) + k*u[i]
    dudt[0] = 0 
    return dudt


uinit = np.zeros(2*m)
uinit[:m] = 1

time = np.linspace(0, dt*snaps, snaps+1)

sol = solve_ivp(deriv, (0, dt*snaps), uinit, method='LSODA', t_eval=time, atol=1e-12, rtol=1e-12)

x_points = np.linspace(0, L, m)

plt.plot(x_points, sol.y[:m, -1], x_points, sol.y[m:, -1])

plt.show()
