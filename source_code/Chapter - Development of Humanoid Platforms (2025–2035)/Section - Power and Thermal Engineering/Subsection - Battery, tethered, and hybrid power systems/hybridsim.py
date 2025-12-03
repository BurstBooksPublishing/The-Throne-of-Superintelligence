import math, numpy as np

# -- Mission parameters (example) --
dt = 1.0                         # timestep sec
T = 3600                         # mission seconds
timesteps = int(T/dt)
P_profile = np.full(timesteps, 200.0)   # W baseline
P_profile[100:120] += 800.0      # actuation burst

# -- Battery spec --
E_batt_wh = 1000.0               # Wh usable
s_eff = 0.95                     # discharge efficiency
SoC = 1.0                        # fraction
C_rate_max = 2.0                 # max discharge 1/h

# -- Tether spec --
tether_available = np.zeros(timesteps, bool)
tether_available[0:timesteps] = True    # example: tether available entire mission
P_tether_max = 1200.0           # W delivered by tether

# -- Thermal limits --
T_junction = 60.0               # degC limit
thermal_capacity = 500.0        # J/K thermal mass
T_env = 25.0
temp = T_env

# -- Simulation loop --
soc = SoC * E_batt_wh
log = []
for k in range(timesteps):
    P_need = P_profile[k]
    # draw from tether first
    P_from_tether = min(P_tether_max if tether_available[k] else 0.0, P_need)
    P_from_batt = P_need - P_from_tether
    # enforce C-rate
    P_from_batt = min(P_from_batt, C_rate_max * E_batt_wh)
    # update SoC
    soc -= (P_from_batt * dt) / 3600.0  # Wh
    # thermal update: electrical losses -> heat (assume 5% losses)
    heat = 0.05 * P_need * dt
    temp += heat / thermal_capacity
    log.append((k*dt, P_from_tether, P_from_batt, soc, temp))
    if soc <= 0.0:
        raise SystemExit("Battery depleted at t={}".format(k*dt))
    if temp > T_junction:
        raise SystemExit("Thermal limit exceeded at t={}".format(k*dt))
# print final state
print("Final SoC (Wh):", soc)