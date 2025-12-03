import numpy as np
from sklearn.linear_model import LinearRegression

# synthetic data: compute C (FLOPs) and validation loss L
C = np.array([1e12, 3e12, 1e13, 3e13, 1e14])  # compute budget samples
L = np.array([2.5, 2.1, 1.9, 1.75, 1.68])     # observed loss

# fit log-log: L - L_inf = k * C^{-alpha} => log(L-Linf) = log k - alpha log C
L_inf = 1.6  # estimated irreducible loss (small sample)
y = np.log(L - L_inf)
X = np.log(C).reshape(-1,1)
reg = LinearRegression().fit(X, y)
alpha = -reg.coef_[0]     # exponent
k = np.exp(reg.intercept_)# prefactor

# predict loss and optimal N assuming C = k_const * N * D and N ~ D => N_opt ~ sqrt(C)
k_const = 1e-9  # FLOPs per (param * token) placeholder
def predict_N_opt(C_budget):
    return np.sqrt(C_budget / k_const)

C_test = 5e14
N_opt = predict_N_opt(C_test)
print(f"fitted alpha={alpha:.3f}, predicted N_opt={N_opt:.2e}")  # prints result