import numpy as np
import torch
import torch.nn as nn

# world_model: must implement step(state, action)->(next_state, reward, done)
# heuristic: PyTorch model returning (value, logvar) for uncertainty-aware pruning

class HeuristicNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,128), nn.ReLU(), nn.Linear(128,2))
    def forward(self,x):
        out = self.net(x)
        return out[:,0], out[:,1]  # value, log-variance

def resource_aware_mcts(root_state, world_model, heuristic, budget_compute, lambda_thresh):
    # Node stores (state, Q, N); simple flat search for clarity
    nodes = [{'state': root_state, 'Q':0.0, 'N':0}]
    compute_used = 0.0
    best_action = None
    while compute_used < budget_compute:
        node = nodes[0]  # expand root children iteratively for example
        # propose actions (world_model provides affordances)
        actions = world_model.affordances(node['state'])
        best_delta = -np.inf
        best_pair = None
        for a in actions:
            s_next, r, done = world_model.step(node['state'], a)
            s_tensor = torch.from_numpy(s_next).float().unsqueeze(0)
            v, logvar = heuristic(s_tensor)
            v = v.item()
            # marginal expected improvement if we expand this action
            expected_gain = r + v - node['Q']
            # per-compute cost estimate (simple constant here)
            cost = world_model.compute_cost(a)
            score = expected_gain / (cost + 1e-6)
            if score > best_delta:
                best_delta = score
                best_pair = (a, s_next, r, cost)
        # pruning rule: stop if best marginal improvement per compute is low
        if best_delta < lambda_thresh:
            break
        # otherwise commit compute to expand chosen action
        a, s_next, r, cost = best_pair
        compute_used += cost
        # simple backup
        node['Q'] = max(node['Q'], r + v)
        node['N'] += 1
        # record best action so far
        best_action = a
    return best_action

# Example instantiation commented out (requires concrete world_model and data).