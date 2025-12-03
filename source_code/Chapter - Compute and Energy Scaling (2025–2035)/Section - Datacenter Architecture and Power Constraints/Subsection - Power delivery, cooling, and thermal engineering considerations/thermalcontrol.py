import time, math, random

# --- simulated sensors/APIs (replace in production) ---
def read_sensors():
    # returns (inlet_C, outlet_C, rack_power_kW, coolant_flow_Ls)
    base_T = 20.0
    rack_power = 500.0  # kW total for pod
    # simulate small fluctuations
    inlet = base_T + random.uniform(-0.5,0.5)
    outlet = inlet + (rack_power/500.0)*5.0 + random.uniform(-0.5,0.5)  # scaled delta-T
    flow = 24.0 + random.uniform(-1.0,1.0)  # L/s ~ mass flow proxy
    return inlet, outlet, rack_power, flow

def set_coolant_flow(new_flow):
    # actuator stub: set pump speed; returns acknowledged flow
    return max(0.0, min(100.0, new_flow))

def migrate_batches(fraction):
    # orchestration stub: migrate fraction of batch jobs
    return f"migrated {fraction*100:.0f}%"

# --- fusion/state estimation ---
alpha = 0.3  # EWMA smoothing
smoothed_deltaT = None

def update_state(inlet, outlet, power_kW, flow_Ls):
    global smoothed_deltaT
    deltaT = outlet - inlet
    if smoothed_deltaT is None:
        smoothed_deltaT = deltaT
    else:
        smoothed_deltaT = alpha*deltaT + (1-alpha)*smoothed_deltaT
    return {'deltaT':deltaT, 'deltaT_smooth':smoothed_deltaT,
            'power_kW':power_kW, 'flow_Ls':flow_Ls}

# --- control policy (predictive safety) ---
TARGET_DT = 5.0  # K
MAX_DT = 8.0     # emergency threshold
FLOW_KP = 2.0    # proportional control gain for pump (L/s per K)

for step in range(60):  # one-minute cadence simulation
    inlet, outlet, power, flow = read_sensors()
    state = update_state(inlet, outlet, power, flow)
    # simple predictive trend (linear estimate)
    trend = 0.1*(state['deltaT'] - state['deltaT_smooth'])
    predicted_dt = state['deltaT_smooth'] + trend*5.0
    # decide actions
    if predicted_dt > TARGET_DT:
        desired_flow = flow + FLOW_KP*(predicted_dt - TARGET_DT)
        flow = set_coolant_flow(desired_flow)
        action = f"increase_flow to {flow:.1f} L/s"
    else:
        action = "no_change"
    if predicted_dt > MAX_DT:
        # emergency: shed fraction of batch load
        mig = migrate_batches(0.5)
        action += f"; emergency_migration: {mig}"
    # telemetry/logging
    print(f"t={step}s dt={state['deltaT']:.2f}K sm={state['deltaT_smooth']:.2f}K pred={predicted_dt:.2f}K action={action}")
    time.sleep(0.1)  # simulate wait