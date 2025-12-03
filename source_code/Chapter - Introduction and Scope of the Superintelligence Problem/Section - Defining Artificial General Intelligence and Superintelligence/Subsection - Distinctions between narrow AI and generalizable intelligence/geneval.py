import numpy as np

# -- stubs: replace with real sensors, LLMs, controllers
def simulate_task(task_id, policy):  # returns success probability
    np.random.seed(task_id)           # deterministic per task
    base = 0.5 + 0.1*(policy['skill']-1)  # skill influences base
    noise = np.random.randn()*0.05
    return float(np.clip(base + noise, 0.0, 1.0))

def train_policy_on_family(family_ids):
    # simple meta-parameter: skill level learned from family
    return {'skill': 1 + len(family_ids)/len(all_train)}  # proxy

# define task families
all_train = list(range(0,50))      # narrow training tasks
all_unseen = list(range(50,80))    # evaluation tasks (novel)

# train and evaluate
policy = train_policy_on_family(all_train)
P_train = np.mean([simulate_task(t, policy) for t in all_train])
G_unseen = np.mean([simulate_task(t, policy) for t in all_unseen])
delta = P_train - G_unseen

print(f"P_train={P_train:.3f}, G_unseen={G_unseen:.3f}, Delta={delta:.3f}")