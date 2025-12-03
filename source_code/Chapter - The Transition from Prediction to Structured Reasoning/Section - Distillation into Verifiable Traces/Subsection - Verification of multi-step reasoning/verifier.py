import math, heapq

# Example step: {'assertion':..., 'evidence':..., 'weight':w, 'cost':c}
def local_check(step):
    # cheap syntactic/evidence replay check (True/False)
    return 'evidence' in step and step['evidence'] is not None

def semantic_score(step):
    # inexpensive proxy for P(a_i | e_i) (0..1)
    return min(1.0, 0.1 + 0.9 * (len(str(step['assertion'])) % 7) / 6.0)

def expensive_formal_check(step):
    # placeholder for formal prover or simulator rollouts
    # return (verified_bool, verification_value)
    return True, semantic_score(step) * 0.95

def verify_trace(trace, budget):
    # stage 1: fast checks, collect candidates for expensive checks
    candidates = []
    logV = 0.0
    for i, step in enumerate(trace):
        if not local_check(step): 
            step['flag'] = 'broken_evidence'; continue
        p = semantic_score(step)
        c = step.get('cost', 1.0)
        w = step.get('weight', 1.0)
        # score contribution if not fully verified
        logV += math.log(max(1e-6, p)) * (step.get('alpha',1.0))
        # push candidate (priority = utility per cost)
        heapq.heappush(candidates, (-w/c, i, step))
    # stage 2: greedy expensive checks under budget
    remaining = budget
    while candidates and remaining > 0:
        _, i, step = heapq.heappop(candidates)
        cost = step.get('cost',1.0)
        if cost <= remaining:
            ok, v = expensive_formal_check(step)  # expensive op
            remaining -= cost
            logV += math.log(max(1e-6, v)) * 0.5  # additional confidence boost
            if not ok: step['flag'] = 'formal_failed'
    return math.exp(logV), trace

# small runnable example
trace = [{'assertion':'x>0','evidence':'sensorA', 'weight':2.0,'cost':3},
         {'assertion':'y==f(x)','evidence':'simB', 'weight':5.0,'cost':5}]
score, annotated = verify_trace(trace, budget=5)
print('verif_score', score, 'annotated', annotated)  # inline comment: simple demo