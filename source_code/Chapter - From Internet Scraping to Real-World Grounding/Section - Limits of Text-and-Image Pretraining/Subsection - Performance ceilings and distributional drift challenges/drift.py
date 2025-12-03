import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# X_train, X_live are feature matrices from pretrained encoder; y labels unused here.
# Build domain classifier: 0=train, 1=live
X = np.vstack([X_train, X_live])
y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_live))])
clf = LogisticRegression(max_iter=200).fit(X, y)         # density-ratio via classifier
p_live = clf.predict_proba(X)[:,1]
# density ratio for train samples: w = p_live / (1 - p_live) * (n_train/n_live) if priors equal adjust
p_model = p_live[:len(X_train)]                           # probabilities for train subset
w = p_model / (1.0 - p_model + 1e-12)                     # importance weights (unstabilized)
# compute N_eff
N_eff = (w.sum()**2) / (np.square(w).sum() + 1e-12)
# domain separability diagnostic
auc = roc_auc_score(y, p_live)
print(f"AUC={auc:.3f}, N_eff={N_eff:.1f}, N_train={len(X_train)}")
# emit alerts if AUC high or N_eff low (external governance layer acts on alerts).