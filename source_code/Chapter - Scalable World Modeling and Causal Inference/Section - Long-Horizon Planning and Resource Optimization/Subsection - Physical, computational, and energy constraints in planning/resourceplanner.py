import math, random, time

# --- system models (simplified) ---
def move_energy(velocity): return 0.1*velocity**2 + 0.5  # J per step
def comp_energy(flop_seconds): return 5.0 * flop_seconds  # J per second
def thermal_update(thermal, comp_seconds): return thermal + 2.0*comp_seconds - 0.5  # degC

# --- synthetic perception/fusion ---
def perceive(): return {'wind': random.uniform(-2,2), 'obs': random.random()<0.05}
def fuse(obs): return {'state': (random.random(), random.random())}  # placeholder

# --- EVC estimator: delta reward per compute-second (simple proxy) ---
def estimate_evc(problem_difficulty, depth):
    base_gain = max(0.0, 1.0 - 0.2*depth)  # diminishing returns model
    return base_gain / (0.1 + 0.5*problem_difficulty)

# --- bounded planner (stub) ---
def plan(state, depth):
    # returns action and planner compute seconds used (proportional to depth)
    comp_seconds = 0.05 * (2**depth)  # exponential cost with depth
    # action: choose velocity proportional to depth but noisy
    action = min(5.0, 1.0 + 0.5*depth + random.gauss(0,0.1))
    return action, comp_seconds

# --- runtime loop ---
battery = 1000.0  # J
thermal = 40.0    # degC
C_max = 1.0       # max comp seconds per real-time step
for step in range(100):
    sensors = perceive()
    fused = fuse(sensors)
    difficulty = 1.0 if sensors['obs'] else 0.3  # higher when obstacle present

    # choose depth by EVC under compute budget and thermal constraints
    best = {'depth':0,'evc':0.0,'comp':0.0}
    for depth in range(0,5):
        evc = estimate_evc(difficulty, depth)
        comp_est = 0.05*(2**depth)
        if comp_est > C_max: break  # enforce compute capacity
        if thermal + 2.0*comp_est > 85.0: break  # enforce thermal safety
        if evc > best['evc']:
            best = {'depth':depth,'evc':evc,'comp':comp_est}

    # plan with selected depth
    action, comp_used = plan(fused['state'], best['depth'])
    # update energy and thermal
    e_move = move_energy(action)
    e_comp = comp_energy(comp_used)
    battery -= (e_move + e_comp)
    thermal = thermal_update(thermal, comp_used)

    # execute action (placeholder)
    # send_actuator_command(action)

    # telemetry and safety checks
    if battery < 50.0:
        # reduce compute aggressively to conserve energy
        C_max = 0.2
    if thermal > 80.0:
        # forced cool-down: zero compute for next step
        C_max = 0.0

    time.sleep(0.01)  # simulate real-time step