import numpy as np
# Simple expert callable interface (returns logits and aux info)
class Expert:
    def __init__(self, cost): self.cost = cost
    def __call__(self, x): return np.tanh(x.sum())  # placeholder

def gating_logits(x, num_experts):
    # cheap gating producing logits (shape: num_experts)
    return np.linspace(1, num_experts, num_experts) * (x.mean()+0.1)

def select_experts(logits, costs, budget, temperature=1.0, topk=2):
    # softmax then top-k; enforce budget by dropping lowest-utility experts
    probs = np.exp(logits/temperature)
    probs /= probs.sum()
    order = np.argsort(-probs)  # descending
    selected, cumcost = [], 0.0
    for idx in order[:topk]:
        if cumcost + costs[idx] <= budget:
            selected.append(idx); cumcost += costs[idx]
    return selected

# pipeline
x = np.random.randn(128)                 # fused perception-retrieval vector
num_experts = 8
experts = [Expert(cost=1.0+i*0.5) for i in range(num_experts)]
costs = [e.cost for e in experts]
logits = gating_logits(x, num_experts)
initial_sel = select_experts(logits, costs, budget=3.0, temperature=0.7, topk=3)
outputs = [experts[i](x) for i in initial_sel]
ensemble_mean = np.mean(outputs)         # fused output
uncertainty = np.std(outputs)            # simple uncertainty proxy

if uncertainty > 0.3:
    # fallback: expand selection respecting a higher budget
    expanded_sel = select_experts(logits, costs, budget=6.0, temperature=1.5, topk=6)
    outputs = [experts[i](x) for i in expanded_sel]
    ensemble_mean = np.mean(outputs)     # refined output
# ensemble_mean used downstream for action