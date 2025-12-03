import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# simple in-memory KG and embeddings
KG = {'concepts': [], 'embeddings': []}
tau_novel = 0.7  # novelty threshold
p_accept = 0.8   # acceptance probability threshold

def fuse_embeddings(embs, weights=None):
    w = np.ones(len(embs)) if weights is None else np.array(weights)
    vec = (w[:,None]*np.array(embs)).sum(axis=0)
    return vec / np.linalg.norm(vec)

def novelty_score(vec, bank):
    if not bank: return 0.0
    sims = cosine_similarity([vec], bank)[0]
    return sims.max()

def embodied_probe_execute(action_fn, probe_inputs):
    # action_fn executes minimal interaction; returns observation embedding
    return action_fn(probe_inputs)

def pipeline_step(modal_embs, action_fn):
    cand = fuse_embeddings(modal_embs)
    s = novelty_score(cand, KG['embeddings'])
    if s < tau_novel:
        obs = embodied_probe_execute(action_fn, cand)   # active test
        updated = fuse_embeddings([cand, obs], weights=[1.0, 1.0])
        # naive accept/reject based on similarity to itself after probe
        p = novelty_score(updated, KG['embeddings'])
        if p < p_accept:
            KG['concepts'].append({'vec': updated.tolist(), 'meta': {'probes':1}})
            KG['embeddings'].append(updated)
            return "committed"
    return "merged_or_ignored"

# Example action function (simulation) returning a perturbed embedding
def example_action(cand):
    return cand + 0.01 * np.random.randn(*cand.shape)

# run step with dummy modal embeddings
modal = [np.random.randn(256) for _ in range(3)]
modal = [v/np.linalg.norm(v) for v in modal]
print(pipeline_step(modal, example_action))  # -> "committed" or "merged_or_ignored"