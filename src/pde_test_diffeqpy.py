import numpy as np
import matplotlib.pyplot as plt
from diffeqpy import de

m = 60
snaps = 600

L = 2
dx = L / m
dt = 0.1

v0 = 0.1 
k = 0.2


def dCdt(dC, C, p, t):
    for i in range(1, m):
        dC[i] = (-v0 * ((C[i] - C[i-1]) / (dx))) - k*C[i]
        dC[i + m] = (-v0 * ((C[i + m] - C[i + m - 1]) / (dx))) + k*C[i]
    return dC


uinit = np.zeros(2*m)
uinit[:m] = 1

t_span = (0, dt*snaps)


prob = de.ODEProblem(dCdt, uinit, t_span)
sol = de.solve(prob, de.Tsit5(), saveat=0.01)

time = sol.t
C_sol = sol.u
x_points = np.linspace(0, L, m)

plt.plot(x_points, C_sol[100][m:], x_points, C_sol[100][:m])
plt.show()
