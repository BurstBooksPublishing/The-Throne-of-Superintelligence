import random
from statistics import mean

# simple agent classes with produce(), verify() methods
class Proposer:
    def __init__(self, id, bias=0.0):
        self.id = id; self.bias = bias
    def produce(self, state):                      # generate candidate plan + score
        plan = f"move_to_{state['target']}_{self.id}"
        score = max(0.0, 1.0 - abs(self.bias - state['noise']))
        return {'plan': plan, 'score': score, 'rationale': f"b{self.bias}"}

class Verifier:
    def __init__(self, id, robustness=1.0):
        self.id = id; self.robustness = robustness
    def verify(self, candidate, state, adversary=False):  # adversarial check
        base = candidate['score'] * self.robustness
        if adversary:                                    # targeted perturbation
            base *= 0.5
        # return pass probability and objection text
        return {'pass_prob': base, 'objection': None if base>0.6 else "low_conf"}

# pipeline: generate candidates, run verifiers, aggregate
def internal_debate(state, proposers, verifiers, adversary_rate=0.2, quorum=0.6):
    candidates = [p.produce(state) for p in proposers]                  # proposal stage
    results = []
    for c in candidates:
        scores = []
        for v in verifiers:
            adv = random.random() < adversary_rate                       # simulate adversarial test
            r = v.verify(c, state, adversary=adv)
            scores.append(r['pass_prob'])
        aggregate = mean(scores)                                        # simple aggregation
        results.append({'candidate': c, 'verifier_scores': scores, 'aggregate': aggregate})
    # arbiter: select candidate above quorum threshold
    winner = max(results, key=lambda r: r['aggregate'])
    if winner['aggregate'] >= quorum:
        return {'action': winner['candidate']['plan'], 'confidence': winner['aggregate']}
    return {'action': 'HUMAN_REVIEW', 'confidence': winner['aggregate']}

# run simulation
state = {'target':'A', 'noise': 0.1}
proposers = [Proposer(i, bias=0.1*i) for i in range(3)]
verifiers = [Verifier(i, robustness=0.9 - 0.1*(i%2)) for i in range(5)]
print(internal_debate(state, proposers, verifiers))