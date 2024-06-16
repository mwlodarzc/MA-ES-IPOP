import numpy as np
import time

class MAES_IPOP:
    def __init__(self, x0, sigma, options):
        self.start_time = time.time()

        self.sigma = sigma
        self.initial_sigma = sigma
        print(sigma)
        self.maxfevals = options.get('maxfevals', 10000)
        self.N = len(x0)
        self.eval_count = 0
        self.iteration = 0

        # Restart tracking
        self.restart_count = 0  # Track number of restarts

        # Algorithm parameters
        self.lambda_ = int(4 + np.floor(3 * np.log(self.N)))   # Offspring size

        self.update_population_parameters()  # Use a method to update population parameters

        self.y = np.array(x0, dtype=np.float64)
        self.M = np.eye(self.N, dtype=np.float64)  # Initial transformation matrix (identity)
        self.s = np.zeros(self.N, dtype=np.float64)  # Evolution path for step size control
        self.psigma = 0

        self.best_y = None
        self.best_score = np.inf

    def update_population_parameters(self):
        self.mu = int(np.floor(self.lambda_/2))  # Population size
        self.weights = np.array([np.log(self.mu + 1) - np.log(i) for i in range(1, self.mu + 1)], dtype=np.float64)
        self.weights /= np.sum(self.weights)  # Normalize weights
        self.mu_eff = 1 / np.sum(self.weights**2)  # Effective population size

        self.cs = (self.mu_eff + 2) / (self.N + self.mu_eff + 5)  # Cumulation for step size control
        self.dsigma = 1 + self.cs + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.N + 1)) - 1)  # Damping for step size
        self.c1 = 2 / ((self.N + 1.3)**2 + self.mu_eff)  # Learning rate for rank-one update
        self.cw = np.min([1-self.c1, 2*(self.mu_eff-2+1/self.mu_eff) / (self.mu_eff + (self.N+2)**2)])  # Learning rate for rank-mu update

    @staticmethod
    def weighted_sum(values, weights):
        weighted_values = values * weights[:, np.newaxis, np.newaxis]
        return np.sum(weighted_values, axis=0)

    @staticmethod
    def weighted_average(values, weights):
        return np.sum(values * weights[:, np.newaxis], axis=0) / np.sum(weights)

    def stop(self):
        return self.eval_count >= self.maxfevals

    def ask(self):
        z = np.random.randn(self.lambda_, self.N)
        d = z @ self.M.T
        return z, d, self.y + self.sigma * d

    def tell(self, z, d, solutions, scores):
        self.eval_count += len(scores)
        if np.min(scores) < self.best_score:
            self.best_score = np.min(scores)
            self.best_y = solutions[np.argmin(scores)]

        best_indices = np.argsort(scores)[:self.mu]
        z_w = __class__.weighted_average(z[best_indices], self.weights)
        d_w = __class__.weighted_average(d[best_indices], self.weights)

        self.y += self.sigma * d_w
        self.s = (1 - self.cs) * self.s + np.sqrt(self.mu_eff * self.cs * (2 - self.cs)) * z_w

        outer_products = np.array([np.outer(z_i, z_i) for z_i in z[best_indices]])
        weighted_outer = __class__.weighted_sum(outer_products, self.weights)

        self.M = self.M @ (np.eye(self.N) + self.c1 / 2 * (np.outer(self.s, self.s) - np.eye(self.N)) +
                           self.cw / 2 * (weighted_outer - np.eye(self.N)))
        self.sigma *= np.exp((self.cs / 2) * ((np.linalg.norm(self.s)**2 / self.N)-1))

        self.iteration += 1

        # Check restart condition
        if self.check_restart_condition():
            self.restart()

    def check_restart_condition(self):
            # Check if sigma falls below a threshold
            if self.sigma < 1e-12:
                return True
            
            # No Improvement in Best Objective Function Values
            if self.iteration > (10 + np.ceil(30 * self.N / self.lambda_)):
                recent_best_values = [np.min(scores[-(i+1)]) for i in range(10 + int(np.ceil(30 * self.N / self.lambda_)))]
                if np.max(recent_best_values) - np.min(recent_best_values) == 0:
                    return True

            # Small Standard Deviation
            if np.all(np.std(self.M) < 1e-12) and np.linalg.norm(self.s) < 1e-12:
                return True

            # No Effective Axis Change
            eigenvalues, eigenvectors = np.linalg.eigh(self.M)
            for i in range(len(eigenvalues)):
                perturbation = 0.1 * self.sigma * np.sqrt(eigenvalues[i]) * eigenvectors[:, i]
                if np.linalg.norm(self.y + perturbation - self.y) < 1e-10:
                    return True

            # No Effective Coordinate Change
            for i in range(self.N):
                perturbation = 0.2 * self.sigma
                if np.linalg.norm(self.y[i] + perturbation - self.y[i]) < 1e-10:
                    return True

            # Condition Number of Covariance Matrix
            cond_number = np.linalg.cond(self.M)
            if cond_number > 1e14:
                return True

            return False

    def restart(self):
        self.restart_count += 1
        self.lambda_ *= 2  # Double the population size
        self.update_population_parameters()  # Update population parameters with new lambda

        self.sigma = self.initial_sigma  # Reset sigma
        self.y = np.random.uniform(-5, 5, self.N)  # Reset starting point, could be problem-specific
        self.M = np.eye(self.N, dtype=np.float64)  # Reset transformation matrix
        self.s = np.zeros(self.N, dtype=np.float64)  # Reset evolution path

        print(f"Restarting with population size: {self.lambda_}")

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
