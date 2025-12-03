import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# synthetic lab measurement (unknown to agent)
def lab_measure(x):
    return (np.sin(3*x) + 0.1*x**2 + np.random.randn()*0.02).ravel()

# candidate actions (e.g., reagent ratio, temperature)
X_cand = np.linspace(0.0, 1.0, 200).reshape(-1,1)

# initial design
X = np.array([[0.2],[0.8]])
y = lab_measure(X)

gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4).fit(X, y)

def expected_improvement(mu, sigma, y_best, xi=0.01):
    with np.errstate(divide='ignore'):
        Z = (mu - y_best - xi) / sigma
        from scipy.stats import norm
        return (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

budget = 15
for t in range(budget):
    mu, std = gp.predict(X_cand, return_std=True)
    ei = expected_improvement(mu, std, y.max())
    # simple safety filter: exclude extremes (example constraint)
    safe_mask = (X_cand[:,0] > 0.05) & (X_cand[:,0] < 0.95)
    idx = np.argmax(ei * safe_mask)
    x_next = X_cand[idx:idx+1]
    # execute action via planner (stub): here, directly call lab_measure
    y_next = lab_measure(x_next)
    # update dataset and surrogate
    X = np.vstack([X, x_next]); y = np.vstack([y, y_next])
    gp.fit(X, y.ravel())
    # diagnostic print (could be logged to global fabric)
    print(f"iter {t}: x={x_next.ravel()[0]:.3f}, y={y_next.ravel()[0]:.3f}")