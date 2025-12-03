import numpy as np
# environment: hidden state s in {0,1,2}; observations o ~ P(o|s)
P_o_given_s = np.array([[0.8,0.1,0.1],
                        [0.1,0.8,0.1],
                        [0.1,0.1,0.8]])
prior = np.array([1/3,1/3,1/3])

def posterior(o):
    # Bayes update for passive observation o
    numer = prior * P_o_given_s[:,o]
    return numer / numer.sum()

def entropy(p):
    p = np.clip(p,1e-12,1.0)
    return -np.sum(p * np.log(p))

# passive observation
obs = np.random.choice(3, p=P_o_given_s[np.random.choice(3, p=prior)])
post = posterior(obs)
print("Passive entropy:", entropy(post))

# active actions reveal noisy probe with different confusion matrices
actions = {
    0: np.array([[0.9,0.05,0.05],[0.2,0.7,0.1],[0.2,0.1,0.7]]),
    1: np.array([[0.7,0.2,0.1],[0.1,0.85,0.05],[0.2,0.3,0.5]])
}
def expected_entropy_after_action(A):
    P = actions[A]
    expH = 0.0
    for o_prime in range(3):
        # predictive probability of o' marginalizing s
        p_o = (prior * P[:,o_prime]).sum()
        # posterior conditioning on o' (assume we would incorporate prior only here for demo)
        numer = prior * P[:,o_prime]
        p_s = numer / numer.sum()
        expH += p_o * entropy(p_s)
    return expH

for a in actions:
    print("Action",a,"expected entropy",expected_entropy_after_action(a))