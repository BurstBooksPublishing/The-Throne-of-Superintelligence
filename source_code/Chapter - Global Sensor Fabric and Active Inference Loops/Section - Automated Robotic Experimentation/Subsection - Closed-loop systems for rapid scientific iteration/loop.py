import numpy as np
import math
# Mock sensor read (returns noisy yield)
def read_sensor(x):
    true_yield = 0.8*math.exp(-0.1*(x-5)**2)  # hidden ground truth
    return np.clip(true_yield + np.random.normal(0, 0.03), 0.0, 1.0)

# Simple Bayesian posterior over a scalar parameter theta (reaction peak position)
def bayes_update(prior_mu, prior_var, x, y, obs_var=0.03**2):
    # Likelihood: y ~ f(x;theta)+noise. Linearize for demo: assume f ~ Gaussian peak.
    # Here we treat theta as peak location; approximate with local gradient.
    # This is a toy approximator for demonstration.
    grad = (0.8 * (x-5) * 0.1) * math.exp(-0.1*(x-5)**2)
    if abs(grad) < 1e-6:
        return prior_mu, prior_var
    # Gaussian update in parameter space
    lik_var = obs_var / (grad**2)
    post_var = 1.0 / (1.0/prior_var + 1.0/lik_var)
    post_mu = post_var * (prior_mu/prior_var + (prior_mu + (y - 0.8*math.exp(-0.1*(x-5)**2))/grad)/lik_var)
    return post_mu, post_var

# Monte Carlo EIG approximation
def approximate_eig(prior_mu, prior_var, x, n_samples=50):
    ent_prior = 0.5 * math.log(2*math.pi*math.e*prior_var)
    ent_post_sum = 0.0
    for _ in range(n_samples):
        # sample theta from prior, simulate y, compute posterior variance
        theta = np.random.normal(prior_mu, math.sqrt(prior_var))
        y_sim = read_sensor(x)  # for demo, reuse sensor; in practice use generative model
        _, post_var = bayes_update(prior_mu, prior_var, x, y_sim)
        ent_post_sum += 0.5 * math.log(2*math.pi*math.e*post_var)
    ent_post = ent_post_sum / n_samples
    return ent_prior - ent_post  # EIG estimate

# Main loop
prior_mu, prior_var = 5.0, 1.0  # initial belief about optimum location
for t in range(20):
    # propose candidate actions grid, rank by EIG
    candidates = np.linspace(0,10,21)
    scores = [approximate_eig(prior_mu, prior_var, c) for c in candidates]
    x_next = candidates[int(np.argmax(scores))]
    # safety check: disallow extreme temperatures (example constraint)
    if x_next < 0.5 or x_next > 9.5:
        x_next = max(0.5, min(x_next, 9.5))
    y_obs = read_sensor(x_next)  # execute experiment
    prior_mu, prior_var = bayes_update(prior_mu, prior_var, x_next, y_obs)
    print(f"iter {t}: x={x_next:.2f}, y={y_obs:.3f}, mu={prior_mu:.3f}, var={prior_var:.4f}")
    # verification: quick plausibility check
    if prior_var < 1e-3: break