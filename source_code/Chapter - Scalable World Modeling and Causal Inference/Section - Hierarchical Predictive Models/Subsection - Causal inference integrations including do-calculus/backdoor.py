import numpy as np
import pandas as pd

np.random.seed(0)
N = 20000
# U ~ unobserved confounder
U = np.random.binomial(1, 0.3, size=N)
# Z observed proxy (surface roughness)
Z = np.random.normal(loc=U*0.5, scale=1.0, size=N)
# X chosen policy influenced by U (confounding)
p_X = 0.2 + 0.5*U  # higher chance to pick strong grip when U=1
X = np.random.binomial(1, p_X, size=N)
# Y (slip) depends on X and Z and U through noise; structural equation
logit = -1.0 + 1.5*X - 0.8*Z + 0.6*U
prob_Y = 1.0 / (1.0 + np.exp(-logit))
Y = np.random.binomial(1, prob_Y, size=N)

df = pd.DataFrame({'X': X, 'Y': Y, 'Z_bin': pd.qcut(Z, q=5, labels=False)})  # stratify Z

# Back-door adjustment estimate: average P(Y|X=1,Z)*P(Z)
est = 0.0
for z in sorted(df['Z_bin'].unique()):
    dfz = df[df['Z_bin'] == z]
    p_z = len(dfz)/len(df)
    p_y_given_xz = dfz[dfz['X']==1]['Y'].mean() if (dfz['X']==1).any() else dfz['Y'].mean()
    est += p_y_given_xz * p_z
print("Estimated P(Y|do(X=1)) via back-door:", est)
# For comparison, simulated randomized do-intervention
X_do = np.ones(N)  # force X=1
logit_do = -1.0 + 1.5*X_do - 0.8*Z + 0.6*U
prob_Y_do = 1.0 / (1.0 + np.exp(-logit_do))
print("True P(Y|do(X=1)) (simulated):", prob_Y_do.mean())