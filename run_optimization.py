from modules import SUPPORTED_ALGORITHMS

from cocoex import Suite, Observer
import cocopp

import argparse

def run_optimization(algorithm, suite_name, instance, dimensions, seed, result_folder):
    suite = Suite(suite_name, f'instances:{instance}', f'dimensions:{dimensions}')
    observer = Observer(suite_name, f"result_folder:{result_folder}")

    for problem in suite:
        problem.observe_with(observer)

        print(f'dim: {(problem.dimension)}')
        es = SUPPORTED_ALGORITHMS[algorithm](problem.initial_solution, 0.5, {'seed': seed})
        while not es.stop():
            if algorithm == 'MAES':
                z, d, solutions = es.ask()
                es.tell(z, d, solutions, [problem(x) for x in solutions])
            else:
                solutions = es.ask()
                es.tell(solutions, [problem(x) for x in solutions])
            es.disp()

        problem.free()

def main():
    parser = argparse.ArgumentParser(description='Run optimization with COCO suite.')
    parser.add_argument('--algorithm', type=str, choices=SUPPORTED_ALGORITHMS.keys(), required=True, help='Optimization algorithm to use')
    parser.add_argument('--suite', type=str, default='bbob', help='Suite name (default: bbob)')
    parser.add_argument('--instance', type=int, default=1, help='Instance number (default: 1)')
    parser.add_argument('--dimensions', type=int, default=2, help='Dimensions (default: 2)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--result_folder', type=str, default='example_experiment', help='Result folder name (default: example_experiment)')
    parser.add_argument('--run_cocopp', action='store_true', help='Run COCO post-processing (cocopp) after optimization')

    args = parser.parse_args()

    run_optimization(args.algorithm, args.suite, args.instance, args.dimensions, args.seed, args.result_folder)

    if args.run_cocopp:
        cocopp.main(f'exdata/{args.result_folder}')

if __name__ == '__main__':
    main()
