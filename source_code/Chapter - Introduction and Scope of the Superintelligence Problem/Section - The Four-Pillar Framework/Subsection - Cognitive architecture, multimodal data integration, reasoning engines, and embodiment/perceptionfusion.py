import numpy as np

np.random.seed(0)

# Simulated true state
true_state = np.array([10.0, 0.0])  # position, velocity

# Sensor models (vision, lidar) with noise
def sensor_vision(state):
    return state[0] + np.random.normal(0, 0.8)  # noisy position

def sensor_lidar(state):
    return state[0] + np.random.normal(0, 0.3)  # higher precision

# Simple Kalman-like fusion for position only
def fuse_measurements(meas, var):
    # Weighted average by inverse variance
    w = 1.0 / var
    return np.sum(w * meas) / np.sum(w)

# Lightweight rule-based reasoner (placeholder for LLM+retrieval)
class RuleBasedReasoner:
    def decide(self, pos_est):
        if pos_est > 12.0:
            return "move_back"  # safety rule
        if pos_est < 8.0:
            return "move_forward"
        return "hold_position"

# Run loop
reasoner = RuleBasedReasoner()
for t in range(5):
    v = sensor_vision(true_state)
    l = sensor_lidar(true_state)
    pos_est = fuse_measurements(np.array([v, l]), np.array([0.8**2, 0.3**2]))
    action = reasoner.decide(pos_est)
    print(f"t={t}, vision={v:.2f}, lidar={l:.2f}, pos_est={pos_est:.2f}, action={action}")
    # Apply action as a deterministic change for simulation
    if action == "move_forward":
        true_state[0] += 0.5
    elif action == "move_back":
        true_state[0] -= 0.5