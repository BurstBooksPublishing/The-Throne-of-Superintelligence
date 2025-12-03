import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load fused features: visual, depth, proprioceptive; y is success label
X_visual = np.load('X_vis.npy')        # visual embeddings
X_depth  = np.load('X_depth.npy')      # depth features
X_prop   = np.load('X_prop.npy')       # proprioception
y        = np.load('y.npy')

# simple fusion
X = np.concatenate([X_visual, X_depth, X_prop], axis=1)

# hypothesized confounder signal (e.g., lighting proxy estimated from metadata)
confound_proxy = np.load('lighting_est.npy')  # continuous

# estimate propensity p(confound|features) using logistic regression
prop_model = LogisticRegression(max_iter=200)
# binarize proxy for propensity modeling (example)
confound_bin = (confound_proxy > np.median(confound_proxy)).astype(int)
prop_model.fit(X, confound_bin)
p = prop_model.predict_proba(X)[:,1]  # propensity scores

# compute importance weights to decorrelate confounder from outcome
eps = 1e-6
weights = (confound_bin / (p+eps)) + ((1-confound_bin) / (1-p+eps))

# train downstream policy model (weighted), here a random forest as placeholder
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X, y, weights, test_size=0.2, random_state=0)
policy = RandomForestClassifier(n_estimators=200)
policy.fit(X_train, y_train, sample_weight=w_train)  # uses weights to debias

# evaluate invariance across held-out environment splits
v_score = policy.score(X_val, y_val)
print("Validation accuracy (reweighted):", v_score)