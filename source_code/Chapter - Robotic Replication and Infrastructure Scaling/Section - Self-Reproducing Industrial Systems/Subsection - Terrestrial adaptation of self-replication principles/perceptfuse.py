import numpy as np
# simple simulated sensors
def camera_image(): return np.random.rand(480,640,3)  # RGB image
def depth_map(): return np.random.rand(480,640)      # depth in meters
def swir_spectrum(): return np.array([0.2,0.7,0.1])  # simulated material signature

# fusion: extract coarse features, classify feedstock
def fuse_and_classify(img, depth, spectrum):
    # simple features: mean color, mean depth, spectrum vector
    feat = np.concatenate([img.mean(axis=(0,1)), [depth.mean()], spectrum])
    # rule-based classifier for demonstration
    if feat[3] < 0.6 and feat[4] > 0.6: return "recycled_polymer"
    return "unknown_feedstock"

# planning: select fabrication recipe and safe rate
RECIPES = {"recycled_polymer": {"time_min":30, "energy_kwh":2.5}}
def plan_build(feedstock):
    recipe = RECIPES.get(feedstock)
    if recipe is None: return None
    # enforce safety throttle based on mock grid signal
    grid_available = 0.8  # fraction of nominal capacity
    safe_rate = max(0.1, grid_available)  # throttle minimum
    return {"recipe":recipe, "start_delay_s":int(60*(1-safe_rate))}

# action: emit signed intent (simulated)
def emit_action(intent):
    if intent is None:
        print("HOLD: manual review required.")  # operator intervention
        return False
    # simulated signature and actuator command
    command = {"op":"start_job", "params":intent}
    print("COMMAND:", command)
    return True

# loop
img = camera_image(); depth = depth_map(); spec = swir_spectrum()
feedstock = fuse_and_classify(img, depth, spec)           # perception+fusion
intent = plan_build(feedstock)                            # cognition
emit_action(intent)                                       # action