from .maes import MAES
from .maes_ipop import MAES_IPOP

import cma

SUPPORTED_ALGORITHMS = {
    'CMAES': cma.CMAEvolutionStrategy,
    'MAES': MAES,
    'MAES_IPOP': MAES_IPOP
}