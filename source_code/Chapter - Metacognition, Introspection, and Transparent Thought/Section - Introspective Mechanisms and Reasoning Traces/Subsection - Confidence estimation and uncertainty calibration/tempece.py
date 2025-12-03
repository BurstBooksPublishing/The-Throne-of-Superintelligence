import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

# logits: (N,C) numpy array, labels: (N,) ints
def softmax_probs(logits, T=1.0):
    z = torch.from_numpy(logits).float() / float(T)
    return F.softmax(z, dim=1).numpy()

def nll_with_T(T, logits, labels):
    probs = softmax_probs(logits, T[0])
    eps = 1e-12
    nll = -np.log(np.maximum(probs[np.arange(len(labels)), labels], eps)).mean()
    return nll

def temperature_scale(logits, labels):
    init = np.array([1.0])
    res = minimize(nll_with_T, init, args=(logits, labels), bounds=[(1e-3, 100.)])
    return float(res.x[0])

def compute_ece(logits, labels, T=1.0, n_bins=15):
    probs = softmax_probs(logits, T)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    ece = 0.0
    N = len(labels)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if mask.sum() == 0: 
            continue
        acc = (preds[mask] == labels[mask]).mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum()/N) * abs(acc - avg_conf)
    return ece

# Example usage (logits, labels obtained from model + verifier dataset)
# T_opt = temperature_scale(logits, labels)
# ece_before = compute_ece(logits, labels, T=1.0)
# ece_after  = compute_ece(logits, labels, T=T_opt)