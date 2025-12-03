import time
import numpy as np

def read_imu(): return np.random.randn(3)  # replace with real sensor API
def read_camera(): return np.random.randn(480,640,3)  # placeholder
def estimate_state(imu, cam): return np.hstack([imu, cam.mean(axis=(0,1))])  # simple fusion

def uncertainty_score(state): return np.var(state)  # surrogate uncertainty

def exploratory_action(): return {"motor_cmd": np.random.uniform(-1,1,3)}  # policy placeholder

log = []
THRESH = 0.5  # threshold for exploration

for step in range(1000):
    t = time.time()
    imu = read_imu()               # high-rate proprioceptive data
    cam = read_camera()            # lower-rate exteroceptive data
    state = estimate_state(imu, cam)  # fused estimate
    u = uncertainty_score(state)
    if u > THRESH:
        act = exploratory_action()   # trigger active data collection
    else:
        act = {"motor_cmd": np.zeros(3)}  # nominal policy
    # execute act on robot (omitted) and record outcome
    log.append({"t": t, "state": state, "act": act, "uncertainty": u})
    time.sleep(0.01)  # control timestep