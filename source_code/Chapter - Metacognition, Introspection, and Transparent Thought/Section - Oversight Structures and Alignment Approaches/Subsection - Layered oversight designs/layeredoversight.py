import random, hashlib, json

# sensor fusion (simulated): returns confidence and trace
def sensor_fusion():
    c = 0.92  # fused sensor confidence
    t = {"sensor_ok": True}
    return c, t

# planner (simulated LLM): emits plan, confidence, and trace
def planner(state):
    plan = ["approach","grasp","lift"]
    c = 0.80
    t = {"coherence_score": 0.8}
    return plan, c, t

# verifier (simulated Monte Carlo): returns estimated success prob
def verifier(plan):
    successes = sum(random.random() < 0.6 for _ in range(20))
    c = successes / 20.0
    t = {"rollouts": 20, "successes": successes}
    return c, t

# adjudicator: aggregates using Eq. (1) and enforces thresholds
def adjudicate(confidences, traces, weights, tau_auto=0.8, tau_review=0.6):
    prod = 1.0
    for w, c in zip(weights, confidences):
        prod *= (1 - w * c)
    C = 1 - prod
    # log trace with integrity
    log = {"C": C, "confidences": confidences, "traces": traces}
    digest = hashlib.sha256(json.dumps(log).encode()).hexdigest()
    log["digest"] = digest
    print("Aggregate C=", round(C,3))
    if C >= tau_auto:
        return "auto", log
    if C >= tau_review:
        return "review", log
    return "block", log

# pipeline run
s_c, s_t = sensor_fusion()
plan, p_c, p_t = planner(None)
v_c, v_t = verifier(plan)
decision, log = adjudicate([s_c, p_c, v_c], [s_t, p_t, v_t], [0.3,0.4,0.3])
print("Decision:", decision)