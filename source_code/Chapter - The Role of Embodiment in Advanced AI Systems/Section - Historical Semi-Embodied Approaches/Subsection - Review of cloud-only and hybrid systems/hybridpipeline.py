import time, random

# Edge: sensor sampling and local preprocessing
def sample_sensors():
    return {'dist': random.uniform(0,5), 'image_hash': random.randint(0,1000)}

def local_fusion(sensors):
    # simple short-horizon state
    return {'obstacle': sensors['dist'] < 1.0, 'sensor_stamp': time.time()}

# Cloud: heavy reasoning (simulated LLM planner)
def cloud_reasoning(state_snapshot):
    # expensive plan generation simulated by sleep
    time.sleep(0.05)  # cloud compute latency
    if state_snapshot['obstacle']:
        return {'action': 'stop', 'confidence': 0.99, 'plan_id': random.randint(1,1e6)}
    return {'action': 'move_forward', 'confidence': 0.85, 'plan_id': random.randint(1,1e6)}

# Edge safety verification and actuation
def safety_check(plan, local_state):
    # deterministic safety: reject plan that contradicts local sensor
    if local_state['obstacle'] and plan['action'] != 'stop':
        return False
    return plan['confidence'] > 0.7

def actuator_execute(plan):
    print(f"EXECUTE: {plan['action']} (plan {plan['plan_id']})")

# Main loop
for _ in range(10):
    sensors = sample_sensors()                  # edge perception
    local_state = local_fusion(sensors)         # edge fusion
    snapshot = local_state.copy()               # snapshot sent to cloud
    plan = cloud_reasoning(snapshot)            # cloud cognition
    if safety_check(plan, local_state):         # edge verification
        actuator_execute(plan)                  # action
    else:
        print("REJECTED plan; applying safe fallback")
    time.sleep(0.02)                            # control loop cadence