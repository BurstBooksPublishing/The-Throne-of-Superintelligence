import numpy as np

# discrete model: states S, observations O
S = 3; O = 3
# transition p(s'|s,a) shape (A,S,S)
A = 2
P = np.zeros((A,S,S))
# simple deterministic-ish transitions (rows: action)
P[0] = np.array([[0.8,0.1,0.1],
                 [0.1,0.8,0.1],
                 [0.1,0.1,0.8]])
P[1] = np.roll(P[0], 1, axis=1)  # alternative dynamics
# observation likelihood p(o|s) shape (O,S)
lik = np.eye(O) * 0.9 + 0.1 / O
# prior over preferred observations (goal)
pref_o = np.array([0.1,0.1,0.8])  # prefer obs index 2
# current posterior over states q(s)
q_s = np.array([0.6,0.3,0.1])

# enumerate short policies (sequences of two actions)
policies = [np.array([a,b]) for a in range(A) for b in range(A)]
gamma = 1.0  # precision

def expected_free_energy(q_s, policy):
    # forward simulate belief over future states and obs
    q = q_s.copy()
    G = 0.0
    for a in policy:
        # predict next state distribution
        q = q @ P[a].T  # q' = sum_s q(s) p(s'|s,a)
        # predictive obs distribution
        q_o = lik @ q
        # pragmatic term: divergence from preferred obs (negative log-likelihood)
        pragmatic = -np.sum(q_o * np.log(pref_o + 1e-12))
        # epistemic term: KL between predicted state and prior (uniform here)
        prior_s = np.ones(S)/S
        epi = np.sum(q * (np.log(q + 1e-12) - np.log(prior_s)))
        G += pragmatic + epi
    return G

# compute posterior over policies
Gs = np.array([expected_free_energy(q_s, p) for p in policies])
logp = -gamma * Gs
p_pi = np.exp(logp - np.max(logp))
p_pi /= p_pi.sum()

# select and execute first action of sampled policy
idx = np.random.choice(len(policies), p=p_pi)
action = policies[idx][0]
print("Selected action:", action)