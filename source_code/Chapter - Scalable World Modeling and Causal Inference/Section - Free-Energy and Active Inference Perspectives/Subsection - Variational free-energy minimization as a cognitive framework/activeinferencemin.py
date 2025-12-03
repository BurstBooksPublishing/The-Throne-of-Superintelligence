import numpy as np

# generative model parameters (linear)
A = np.array([[1.0]])        # state transition multiplier
B = {0: np.array([[0.0]]),   # action effect (discrete actions)
     1: np.array([[0.5]])}
C = np.array([[1.0]])        # observation matrix
prior_mean = np.array([0.0]) # p(s) mean
prior_var = np.array([[1.0]])# p(s) variance

def free_energy(q_mean, q_var, obs):
    # negative expected log-likelihood + KL to prior (Gaussian closed-form)
    pred_mean = C @ q_mean
    ll = -0.5 * np.sum((obs - pred_mean)**2)  # assume unit observation noise
    kl = 0.5*(np.trace(np.linalg.inv(prior_var)@q_var)
              + (q_mean-prior_mean).T @ np.linalg.inv(prior_var) @ (q_mean-prior_mean)
              - q_mean.size + np.log(np.linalg.det(prior_var)/np.linalg.det(q_var)))
    return -ll + kl

def expected_free_energy(q_mean, q_var, action):
    # predict next-state mean under action; predict obs; compute F
    s_next = A @ q_mean + B[action]       # point prediction
    o_pred = C @ s_next
    q_var_next = q_var                     # simple propagation (placeholder)
    return free_energy(s_next, q_var_next, o_pred)

# initial posterior
q_mean = np.array([0.1])
q_var = np.array([[0.5]])

# observed sensor reading
obs = np.array([0.2])

# perception update: gradient-free single-step approximate update
# here we do a simple Bayes-like adjustment (illustrative)
q_mean = q_mean + 0.1 * (C.T @ (obs - C @ q_mean))  # small correction

# action selection among discrete actions {0,1}
candidates = [0,1]
gvals = [expected_free_energy(q_mean, q_var, a) for a in candidates]
best_action = candidates[int(np.argmin(gvals))]

print("q_mean=", q_mean, "chosen_action=", best_action)