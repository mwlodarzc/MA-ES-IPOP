import numpy as np

def sumx2(x):
    return np.sum(x**2)

def sort_offspring(population, scores):
    return [x for _, x in sorted(zip(scores, population), key=lambda x: x[0])]

def weighted_sum(values, weights):
    weighted_values = values * weights[:, np.newaxis, np.newaxis]
    return np.sum(weighted_values, axis=0)

def weighted_average(values, weights):
    return np.sum(values * weights[:, np.newaxis], axis=0) / np.sum(weights)

def maes(f):
    mu, lambda_ = 10, 100  # Population and offspring sizes

    weights = np.array([np.log(mu + 1) - np.log(i) for i in range(1, mu + 1)])
    weights = weights / np.sum(weights)  # Normalize weights
    mu_eff = 1 / np.sum(weights**2)  # Effective population size
    
    n = 10  # Dimension of the problem
    sigma = 0.3  # Initial step size
    cs = (mu_eff + 2) / (n + mu_eff + 5)  # Cumulation for step size control
    dsigma = 1 + cs + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)  # Damping for step size
    c1 = 2 / ((n + 1.3)**2 + mu_eff)  # Learning rate for rank-one update
    cw = 2 / (mu_eff + 2)  # Learning rate for rank-mu update
    max_generations = 1000  # Termination condition

    y = np.random.randn(n)  # Initial solution
    M = np.eye(n)  # Initial transformation matrix (identity)
    s = np.zeros(n)  # Evolution path for covariance

    for t in range(max_generations):
        # Generate offspring
        z = np.random.randn(lambda_, n)
        d = z @ M.T
        offspring = y + sigma * d
        scores = np.array([f(ind) for ind in offspring]) # evaluation of the solution's fitness

        # Selection and recombination
        best_indices = scores.argsort()[:mu] 
        z_w = weighted_average(z[best_indices], weights)
        d_w = weighted_average(d[best_indices], weights)

        # Offspring Population
        y = y + sigma * d_w

        # Adaptation of the evolution path and covariance matrix
        s = (1 - cs) * s + np.sqrt(mu_eff * cs * (2 - cs)) * z_w

        outer_products = np.array([np.outer(z_i, z_i) for z_i in z[best_indices]])
        weighted_outer = weighted_sum(outer_products, weights)
        M = M @ (np.eye(n) + c1 / 2 * (np.outer(s, s) - np.eye(n)) +
                cw / 2 * (weighted_outer - np.eye(n)))

        # Step size control
        sigma *= np.exp(cs / dsigma * (np.linalg.norm(s) / np.mean(np.abs(np.random.randn(10000))) - 1))

        # Output current best solution and its fitness
        best_score = scores[best_indices[0]]
        print(f"Generation {t}: Best score = {best_score}, y = {y}")

        if best_score < 1e-10:  # Convergence criterion (can be adjusted)
            break

if __name__ == "__main__":
    maes(sumx2)
