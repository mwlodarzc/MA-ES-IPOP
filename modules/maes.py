import numpy as np
import time

class MAES:
    def __init__(self, x0, sigma, options):
        self.start_time = time.time()
        self.y = np.array(x0)
        self.sigma = sigma
        self.maxfevals = options.get('maxfevals', 10000)
        self.dimension = len(x0)
        self.eval_count = 0
        self.iteration = 0

        # Algorithm parameters
        self.mu = int(4 + np.floor(3 * np.log(self.dimension)))  # Population size
        self.lambda_ = 4 * self.mu  # Offspring size
        self.weights = np.array([np.log(self.mu + 1) - np.log(i) for i in range(1, self.mu + 1)])
        self.weights /= np.sum(self.weights)  # Normalize weights
        self.mu_eff = 1 / np.sum(self.weights**2)  # Effective population size

        self.cs = (self.mu_eff + 2) / (self.dimension + self.mu_eff + 5)  # Cumulation for step size control
        self.dsigma = 1 + self.cs + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dimension + 1)) - 1)  # Damping for step size
        self.c1 = 2 / ((self.dimension + 1.3)**2 + self.mu_eff)  # Learning rate for rank-one update
        self.cw = 2 / (self.mu_eff + 2)  # Learning rate for rank-mu update

        self.M = np.eye(self.dimension)  # Initial transformation matrix (identity)
        self.s = np.zeros(self.dimension)  # Evolution path for covariance
        self.best_y = None
        self.best_score = np.inf

    def stop(self):
        return self.eval_count >= self.maxfevals

    def ask(self):
        z = np.random.randn(self.lambda_, self.dimension)
        d = z @ self.M.T
        return self.y + self.sigma * d

    def tell(self, solutions, scores):
        self.eval_count += len(scores)
        if np.min(scores) < self.best_score:
            self.best_score = np.min(scores)
            self.best_y = solutions[np.argmin(scores)]

        best_indices = np.argsort(scores)[:self.mu]
        z = (solutions - self.y) / self.sigma
        z_w = np.sum(z[best_indices] * self.weights[:, np.newaxis], axis=0) / np.sum(self.weights)
        d_w = np.sum((z @ self.M.T)[best_indices] * self.weights[:, np.newaxis], axis=0) / np.sum(self.weights)

        self.y += self.sigma * d_w
        self.s = (1 - self.cs) * self.s + np.sqrt(self.mu_eff * self.cs * (2 - self.cs)) * z_w
        outer_products = np.array([np.outer(z_i, z_i) for z_i in z[best_indices]])
        weighted_outer = np.sum(outer_products * self.weights[:, np.newaxis, np.newaxis], axis=0)
        self.M = self.M @ (np.eye(self.dimension) + self.c1 / 2 * (np.outer(self.s, self.s) - np.eye(self.dimension)) +
                           self.cw / 2 * (weighted_outer - np.eye(self.dimension)))

        try:
            step_size_control = self.cs / self.dsigma * (np.linalg.norm(self.s) / np.mean(np.abs(np.random.randn(10000))) - 1)
            step_size_control = np.clip(step_size_control, -100, 100)  # Prevent overflow
            self.sigma *= np.exp(step_size_control)
        except OverflowError:
            print("Overflow encountered in step size control.")
            self.sigma *= np.exp(np.clip(step_size_control, -100, 100))

        self.iteration += 1

    def disp(self):
        elapsed_time = time.time() - self.start_time
        min_std = np.min(np.sqrt(np.diag(self.M)))
        max_std = np.max(np.sqrt(np.diag(self.M)))

        try:
            eigenvalues = np.linalg.eigvals(self.M)
            axis_ratio = np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)  # Prevent division by zero
        except np.linalg.LinAlgError:
            axis_ratio = float('inf')
        
        print(f"   {self.iteration:4d}  {self.eval_count:6d} {self.best_score:23.10e} {axis_ratio:7.1e} {self.sigma:7.2e}  {min_std:.1e} {max_std:.1e} {int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}.0")
