import numpy as np
import matplotlib.pyplot as plt

# params
a,b,h,gamma,delta,eta = 0.5, 0.2, 10.0, 0.05, 0.02, 0.01
kappa, O = 0.0, 0.0  # oversight damping
dt, T = 0.01, 200.0
steps = int(T/dt)

def step_cx(C,X):
    phi = kappa*O*C                  # oversight damping
    dC = a*C*X/(h+X) - gamma*C - phi
    dX = b*X*(1+eta*C) - delta*X
    return dC, dX

# run deterministic
C = np.zeros(steps); X = np.zeros(steps); t = np.linspace(0,T,steps)
C[0], X[0] = 1.0, 5.0
for i in range(1,steps):
    dC,dX = step_cx(C[i-1], X[i-1])
    C[i] = C[i-1] + dC*dt
    X[i] = max(1e-6, X[i-1] + dX*dt)

# stochastic SDE for C only (multiplicative noise)
sigma = 0.15
Cs = np.zeros(steps); Cs[0] = 1.0
for i in range(1,steps):
    r_eff = a*X[i-1]/(h+X[i-1]) - gamma - kappa*O
    dW = np.random.normal(0, np.sqrt(dt))
    Cs[i] = max(1e-6, Cs[i-1] + r_eff*Cs[i-1]*dt + sigma*Cs[i-1]*dW)

# plot
plt.plot(t, C, label='deterministic C')
plt.plot(t, Cs, label='stochastic C', alpha=0.6)
plt.plot(t, X/np.max(X)*np.max(C), '--', label='scaled X')
plt.xlabel('time'); plt.ylabel('metric'); plt.legend(); plt.show()