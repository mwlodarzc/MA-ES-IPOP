"""
Module for Evolutionary Strategy Optimization using the (mu, lambda)-ES algorithm.
"""

import numpy as np

def sumx2(x):
    """
    Example objective function: Sphere function.
    Calculates the sum of squares of all elements in x.
    
    Args:
    x (np.ndarray): Input vector.
    
    Returns:
    float: The sum of squares of x.
    """
    return np.sum(x**2)

def sort_offspring(population, scores):
    """
    Sorts the offspring by their fitness scores in ascending order.
    
    Args:
    population (list): List of candidate solutions.
    scores (list): Corresponding fitness scores of candidates.
    
    Returns:
    list: Sorted list of candidate solutions based on fitness.
    """
    return [x for _, x in sorted(zip(scores, population), key=lambda x: x[0])]

def weighted_sum(values, weights):
    """
    Applies weights to calculate a weighted sum of matrices.
    
    Args:
    values (np.ndarray): Array of matrices.
    weights (np.ndarray): Weights applied to the values.
    
    Returns:
    np.ndarray: Weighted sum of matrices.
    """
    weighted_values = values * weights[:, np.newaxis, np.newaxis]
    return np.sum(weighted_values, axis=0)

def weighted_average(values, weights):
    """
    Calculates a weighted average of vectors.
    
    Args:
    values (np.ndarray): Array of vectors.
    weights (np.ndarray): Weights applied to the vectors.
    
    Returns:
    np.ndarray: Weighted average of vectors.
    """
    return np.sum(values * weights[:, np.newaxis], axis=0) / np.sum(weights)

class EvolutionaryOptimizer:
    """
    Evolutionary Strategy Optimization using the (mu, lambda)-ES algorithm.
    """
    def __init__(self, objective_func, mu=10, lambda_=100, n=10, max_generations=1000):
        self.f = objective_func
        self.mu = mu
        self.lambda_ = lambda_
        self.n = n
        self.max_generations = max_generations

    def run(self):
        """
        Executes the (mu, lambda)-ES algorithm to minimize the objective function.
        
        Returns:
        tuple: Best found solution and its fitness score.
        """
        weights = np.array([np.log(self.mu + 1) - np.log(i) for i in range(1, self.mu + 1)])
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)

        sigma = 0.3
        cs = (mu_eff + 2) / (self.n + mu_eff + 5)
        dsigma = 1 + cs + 2 * max(0, np.sqrt((mu_eff - 1) / (self.n + 1)) - 1)
        c1 = 2 / ((self.n + 1.3)**2 + mu_eff)
        cw = 2 / (mu_eff + 2)

        y = np.random.randn(self.n)
        M = np.eye(self.n)
        s = np.zeros(self.n)

        for t in range(self.max_generations):
            z = np.random.randn(self.lambda_, self.n)
            d = z @ M.T
            offspring = y + sigma * d
            scores = np.array([self.f(ind) for ind in offspring])

            best_indices = scores.argsort()[:self.mu]
            z_w = weighted_average(z[best_indices], weights)
            d_w = weighted_average(d[best_indices], weights)

            y = y + sigma * d_w
            s = (1 - cs) * s + np.sqrt(mu_eff * cs * (2 - cs)) * z_w

            outer_products = np.array([np.outer(z_i, z_i) for z_i in z[best_indices]])
            weighted_outer = weighted_sum(outer_products, weights)
            M = M @ (np.eye(self.n) + c1 / 2 * (np.outer(s, s) - np.eye(self.n)) +
                    cw / 2 * (weighted_outer - np.eye(self.n)))

            sigma *= np.exp(cs / dsigma * (np.linalg.norm(s) / np.mean(np.abs(np.random.randn(10000))) - 1))

            if scores[best_indices[0]] < 1e-10:
                break

        return y, scores[best_indices[0]]

if __name__ == "__main__":
    optimizer = EvolutionaryOptimizer(sumx2)
    best_solution, best_score = optimizer.run()
    print(f"Best score: {best_score}, Best solution: {best_solution}")
