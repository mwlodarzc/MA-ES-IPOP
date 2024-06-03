from .maes import MAES

import cma

SUPPORTED_ALGORITHMS = {
    'CMAES': cma.CMAEvolutionStrategy,
    'MAES': MAES,
}