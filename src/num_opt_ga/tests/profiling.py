import cProfile
import pstats

from ..num_opt_ga import NumericalOptimizationGA
from ..examples.sample_functions import damped_cossine


POP_SIZE = 5000
GENS = 500

if __name__ == "__main__":

    ga = NumericalOptimizationGA(
        search_region=((-1, 1), (-1, 1)),
        function=lambda pos: damped_cossine(*pos),
        pop_size=POP_SIZE,
    )

    with cProfile.Profile() as profiler:
        for _ in range(GENS):
            ga.evolve()

    ps = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME).print_stats(10)
    print("Best position =", ga.best(), sep="\n")
