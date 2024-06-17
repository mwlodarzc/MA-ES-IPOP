import os
import subprocess

# Define your algorithms, suite, instances, dimensions, and seed
algorithms = ['CMAES', 'MAES', 'MAES_IPOP']
suite = 'bbob'
instances = list(range(1,16))  # List of instances
dimensions = [2, 3, 5, 10, 20, 40]
seed = 42
base_result_folder = 'all_experiments'

# Ensure the base result directory exists
if not os.path.exists(base_result_folder):
    os.makedirs(base_result_folder)

for algorithm in algorithms:
    algorithm_folder = os.path.join(base_result_folder, algorithm)
    if not os.path.exists(algorithm_folder):
        os.makedirs(algorithm_folder)
    for dimension in dimensions:
        for instance in instances:
            result_folder = os.path.join(algorithm_folder, f'dim_{dimension}_instance_{instance}')
            cmd = [
                'python', 'run_optimization.py',
                '--algorithm', algorithm,
                '--suite', suite,
                '--instances', str(instance),
                '--dimensions', str(dimension),
                '--seed', str(seed),
                '--result_folder', result_folder
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

print("All experiments completed.")
