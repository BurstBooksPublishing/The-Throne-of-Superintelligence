import math,random
# simple RC thermal model and exponential reliability; adjust mission profile as list of (P,W) tuples
C_th = 500.0            # J/K thermal capacitance
R_th = 0.2              # K/W thermal resistance
T_amb = 298.0           # K ambient
lambda_ref = 1e-6       # 1/s base failure rate at T_ref
T_ref = 313.0           # K reference
Q10 = 2.0

def step(T, P, dt):
    # thermal RC integration; P in W, dt in s
    T_inf = T_amb + P*R_th
    T_new = T_inf + (T - T_inf)*math.exp(-dt/(R_th*C_th))
    return T_new

def failure_rate(T):
    return lambda_ref * (Q10 ** ((T - T_ref)/10.0))

# simulate 1-hour mission with alternating compute bursts
T = T_amb
t = 0.0
dt = 1.0
survived = True
while t < 3600:
    P = 600.0 if (int(t/60) % 2 == 0) else 150.0  # 1-min high, 1-min low
    T = step(T, P, dt)
    lam = failure_rate(T)
    # Monte Carlo stochastic failure check for dt
    if random.random() < 1 - math.exp(-lam*dt):
        survived = False
        break
    t += dt

print("Survived:", survived, "Final T (K):", round(T,1))