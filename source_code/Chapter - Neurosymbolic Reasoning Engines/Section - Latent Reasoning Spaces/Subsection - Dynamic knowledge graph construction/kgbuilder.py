import numpy as np
from math import log, exp

# Simple in-memory KG
class KG:
    def __init__(self):
        self.nodes = {}         # node_id -> {'vec': np.array, 'labels': set}
        self.edges = {}         # (a,b,rel) -> {'logodds': float, 't': int}
        self.next_id = 0

    def add_node(self, vec, label=None):
        nid = f"N{self.next_id}"; self.next_id += 1
        self.nodes[nid] = {'vec': vec, 'labels': set() if label is None else {label}}
        return nid

    def match_node(self, vec, thr=0.8):
        # cosine similarity NN search (linear for clarity)
        best, best_sim = None, -1.0
        for nid, v in self.nodes.items():
            sim = float(np.dot(vec, v['vec']) / (np.linalg.norm(vec)*np.linalg.norm(v['vec'])))
            if sim > best_sim:
                best, best_sim = nid, sim
        return best if best_sim >= thr else None

    def update_edge(self, a, b, rel, s, lam=1.0, time=0):
        # s in [-1,1] evidence; convert to log-odds increment
        key = (a,b,rel)
        prior = self.edges.get(key, {'logodds': 0.0})
        prior_logodds = prior['logodds']
        prior_logodds += lam * s
        self.edges[key] = {'logodds': prior_logodds, 't': time}
        return self.prob_from_logodds(prior_logodds)

    @staticmethod
    def prob_from_logodds(lo): return 1.0/(1.0+exp(-lo))

# Example usage
kg = KG()
v_cam = np.random.randn(128); v_cam /= np.linalg.norm(v_cam)
v_touch = v_cam + 0.05*np.random.randn(128); v_touch /= np.linalg.norm(v_touch)

n_cam = kg.add_node(v_cam, label='object_candidate')
match = kg.match_node(v_touch, thr=0.85)
if match is None:
    n_touch = kg.add_node(v_touch, label='tactile_candidate')
else:
    n_touch = match

p = kg.update_edge(n_cam, n_touch, 'coincident', s=0.9, lam=1.2, time=1)  # update belief
print("Edge probability:", p)  # inline diagnostic