import torch, torch.nn as nn
from z3 import Solver, Bool, Implies, Or, Not, sat, unsat

# Simple mapping of atoms to indices
atoms = ['A','B','C']
atom_to_idx = {a:i for i,a in enumerate(atoms)}

# Small MLP proposer: maps state vector to logits over candidate clauses
class Proposer(nn.Module):
    def __init__(self, n_atoms):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_atoms,32), nn.ReLU(), nn.Linear(32, n_atoms*n_atoms))
    def forward(self, state_vec):
        return self.net(state_vec)  # logits for implications i->j

def make_clause(i,j):
    X = Bool(atoms[i]); Y = Bool(atoms[j])
    return Implies(X, Y)

# Utility: check if goal follows from premises with Z3
def is_proved(premises, goal):
    s = Solver()
    for p in premises: s.add(p)
    # to prove goal, check unsat of premises /\ not(goal)
    s.push(); s.add(Not(goal))
    res = s.check()
    s.pop()
    return res == unsat

# Example: premises = {A -> B, B -> C}; goal = A -> C
proposer = Proposer(len(atoms))
opt = torch.optim.Adam(proposer.parameters(), lr=1e-3)

# encode current premises as one-hot count vector
def state_vector(premises):
    v = torch.zeros(len(atoms))
    for p in premises:
        # rough heuristic: mark antecedent presence
        s = str(p)
        for i,a in enumerate(atoms):
            if a in s.split('->')[0]: v[i]=1.0
    return v

for epoch in range(200):
    # start with base premises
    premises = [make_clause(0,1), make_clause(1,2)]  # A->B, B->C
    goal = make_clause(0,2)  # A->C
    sv = state_vector(premises)
    logits = proposer(sv)
    probs = torch.softmax(logits, dim=0)
    # sample top candidate clause index
    idx = int(torch.argmax(probs).item())
    i = idx // len(atoms); j = idx % len(atoms)
    candidate = make_clause(i,j)
    # verifier check
    if is_proved(premises + [candidate], goal):
        reward = 1.0
        loss = -torch.log(probs[idx]) * reward  # policy gradient surrogate
    else:
        reward = 0.0
        loss = torch.log(probs[idx]) * 0.1  # small penalty for useless proposals
    opt.zero_grad(); loss.backward(); opt.step()
    if reward>0:
        print("Proved with candidate:", atoms[i], "->", atoms[j]); break