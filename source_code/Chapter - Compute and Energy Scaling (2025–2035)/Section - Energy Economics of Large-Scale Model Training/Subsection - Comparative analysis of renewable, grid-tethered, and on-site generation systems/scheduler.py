import numpy as np
# inputs: forecasts (hourly arrays), facility state
price_forecast = np.array(...)    # $/kWh, length H
solar_forecast = np.array(...)    # kW available per hour
battery_soc = 0.8                 # fraction
P_pod = 10e3                      # kW
H = len(price_forecast)
# compute expected marginal cost per hour (grid vs onsite solar+storage)
grid_cost = price_forecast
# estimate onsite dispatchable from solar+battery (simplified)
storage_capacity_kwh = 200e3
max_batt_discharge = storage_capacity_kwh * 0.5  # 50% usable
onsite_available = np.minimum(solar_forecast, P_pod)  # direct solar
# combine: use battery to cover remaining if SOC allows
needed = np.full(H, P_pod)
battery_supply = np.zeros(H)
soc = battery_soc * storage_capacity_kwh
for t in range(H):
    residual = max(0.0, needed[t] - onsite_available[t])
    discharge = min(residual, soc, max_batt_discharge)
    battery_supply[t] = discharge
    soc -= discharge
onsite_effective = onsite_available + battery_supply
# fuse cost: when onsite covers full P_pod, marginal cost = amortized onsite LCOE
onsite_lcoe = 0.08  # $/kWh
marginal_cost = np.where(onsite_effective >= P_pod, onsite_lcoe, grid_cost)
# scheduler decision: start now if mean marginal cost below threshold
threshold = 0.06
if np.mean(marginal_cost) < threshold:
    action = "start_job_now"
elif np.min(price_forecast[:24]) < threshold:
    action = "delay_until_low_price"
else:
    action = "start_with_checkpoints"  # accept risk, enable redundancy
print(action)