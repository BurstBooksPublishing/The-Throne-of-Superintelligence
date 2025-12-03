import numpy as np

np.random.seed(0)
T = 200                       # timesteps
c = 0.1                       # initial capability
alpha0 = 0.05                 # base learning gain
beta = 0.6                    # proposal efficacy
gamma = 0.8                   # diminishing returns
history = []

def perceive(c):              # noisy measurement (fusion)
    return c + np.random.normal(scale=0.01)

def propose(c):               # cognition: propose improvement size
    return np.clip(0.2*(1-c), 0, 1)  # smaller proposals as c grows

def evaluate(proposal, meas): # critic: noisy expected improvement
    expected = beta * proposal * (1 - gamma * meas)
    # conservative bias for safety
    return max(0.0, expected - 0.01)

for t in range(T):
    meas = perceive(c)
    proposal = propose(meas)
    score = evaluate(proposal, meas)
    alpha = alpha0 * (1 + 0.5*c)   # adaptive gain example
    c = c + alpha * score          # enact update
    c = np.clip(c, 0.0, 2.0)       # operational bounds
    history.append(c)

# basic diagnostics
print("final capability:", round(c,4))