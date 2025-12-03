import random

class Constitution:
    def __init__(self, constraints): self.constraints = constraints
    def check(self, state, action):         # return True if allowed
        return all(c(state, action) for c in self.constraints)

# simple perception and fusion (stub)
def perceive():                           # returns raw observations
    return {'obstacle_dist': random.uniform(0, 5)}
def fuse(obs):                            # returns belief state
    return {'min_dist': obs['obstacle_dist']}

# cognition: propose candidate actions with scores
def propose_actions(state):
    actions = [{'name':'move_forward','speed':s} for s in [0.1,0.5,1.0]]
    # score by distance and speed (higher better if safe)
    for a in actions: a['score'] = state['min_dist'] - a['speed']
    return sorted(actions, key=lambda x: -x['score'])

# constraints: no collisions, speed cap
def no_collision(state, action): return state['min_dist'] - action['speed'] > 0.2
def speed_cap(state, action):    return action['speed'] <= 0.8

constitution = Constitution([no_collision, speed_cap])

# loop: select first allowed action
for step in range(10):
    obs = perceive()
    state = fuse(obs)
    for cand in propose_actions(state):
        if constitution.check(state, cand):
            chosen = cand; break
    else:
        chosen = {'name':'stop','speed':0.0}  # default safe action
    print(step, state, chosen)               # actuator command would follow