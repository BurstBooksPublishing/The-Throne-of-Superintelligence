import random, math

# Mock sensor fusion output
def fuseSensors():
    return {"parts":["A","B"], "hazard": True, "torqueLimit": 10}

# Mock retrieval aggregator
def retrieveContext(fused):
    return f"parts={fused['parts']}; hazard={fused['hazard']}; torque={fused['torqueLimit']}"

# Mock LLM: generates plan and a simple trace; injects step error with probability eps
def mockLLM(prompt, eps=0.1):
    # produce plan steps and a trace (strings)
    steps = ["pick A", "move to station", "pick B", "assemble"]
    trace = ["checked torque", "checked clearance", "validated order", "validated fit"]
    # simulate omission error
    if random.random() < eps:
        steps.pop(2)  # drop a step to mimic omission
        trace.pop(2)
    return {"plan": steps, "trace": trace, "confidence": 0.9 - eps}

# Simple symbolic verifier enforcing invariants
def verify(plan, fused):
    # ensure both parts present and torque respected (toy checks)
    okParts = all(p in fused["parts"] for p in ["A","B"])
    okTorque = fused["torqueLimit"] >= 5  # example bound
    hasBoth = ("pick A" in plan) and ("pick B" in plan)
    return okParts and okTorque and hasBoth

# Execution path
fused = fuseSensors()
context = retrieveContext(fused)
out = mockLLM(context, eps=0.15)
if verify(out["plan"], fused):
    print("Action approved:", out["plan"])
else:
    print("Verifier failed; escalate to safe fallback")  # safety gating