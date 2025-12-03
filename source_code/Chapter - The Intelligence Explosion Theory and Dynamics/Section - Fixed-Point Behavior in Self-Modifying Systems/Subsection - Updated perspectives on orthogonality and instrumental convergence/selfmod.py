import random, math

def propose_mod(capability):
    # propose delta proportional to capability exploration
    return capability * (0.1 + 0.2 * random.random())

def verify_safety(new_cap):
    # placeholder: replace by formal check, tests, or sandbox evaluation
    # here we require diminishing-risk property: delta <= 0.5*capability
    return new_cap['delta'] <= 0.5 * new_cap['capability']

def evaluate_value(capability):
    # surrogate value combining goal-consistency and resource utility
    return math.log(1 + capability)

capability = 1.0
safety_passes = 0
for step in range(100):
    delta = propose_mod(capability)
    proposal = {'capability': capability + delta, 'delta': delta}
    if verify_safety(proposal):
        # accept only if value increases and safety verified
        if evaluate_value(proposal['capability']) > evaluate_value(capability):
            capability += delta
            safety_passes += 1
    # decay or operational costs
    capability *= 0.999
print("Final capability:", capability, "Accepted upgrades:", safety_passes)