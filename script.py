import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pend(y, t, b, c):
    return np.array([y[1], -b*y[1] - c*np.sin(y[0])])


b = 0.25
c = 5.0
y0 = np.array([np.pi - 0.1, 0.0])
t = np.linspace(0, 10, 101)
sol = odeint(pend, y0, t, args=(b, c))

def rungekutta1(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i+1] = y[i] + (t[i+1] - t[i]) * f(y[i], t[i], *args)
    return y

sol = rungekutta1(pend, y0, t, args=(b, c))
t2 = np.linspace(0, 10, 1001)
sol2 = rungekutta1(pend, y0, t2, args=(b, c))
t3 = np.linspace(0, 10, 10001)
sol3 = rungekutta1(pend, y0, t3, args=(b, c))

def rungekutta2(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        y[i+1] = y[i] + h * f(y[i] + f(y[i], t[i], *args) * h / 2., t[i] + h / 2., *args)
    return y

t4 = np.linspace(0, 10, 21)
sol4 = rungekutta2(pend, y0, t4, args=(b, c))
t = np.linspace(0, 10, 101)
sol = rungekutta2(pend, y0, t, args=(b, c))
t2 = np.linspace(0, 10, 1001)
sol2 = rungekutta2(pend, y0, t2, args=(b, c))
t3 = np.linspace(0, 10, 10001)
sol3 = rungekutta2(pend, y0, t3, args=(b, c))

plt.title("FIDELITY vs TIME")
plt.xlabel("Time")
plt.ylabel("Fidelity")
plt.plot(t4, sol4[:, 0], label='with 11 points')
plt.plot(t, sol[:, 0], label='with 101 points')
plt.plot(t2, sol2[:, 0], label='with 1001 points')
plt.plot(t3, sol3[:, 0], label='with 10001 points')
plt.legend(loc='best')
plt.grid()
plt.show()






