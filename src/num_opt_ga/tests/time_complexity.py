import logging
import big_o
import numpy as np
from numpy.linalg import norm
from ..num_opt_ga import NumericalOptimizationGA

# Setting up logger
LOG_FORMAT = "%(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger("logger")
logger.setLevel("INFO")


def function(pos):
    return np.exp(-((norm(pos) / 0.5) ** 2))


def evolve_const_gen(pop_size: int) -> None:
    ga = NumericalOptimizationGA(
        search_region=((-1, 1), (-1, 1)),
        function=function,
        bits=32,
        pop_size=pop_size,
    )

    while ga.gen < 100:
        ga.evolve()


def evolve_const_pop(gens: int) -> None:
    ga = NumericalOptimizationGA(
        search_region=((-1, 1), (-1, 1)),
        function=function,
        bits=32,
        pop_size=100,
    )

    while ga.gen < gens:
        ga.evolve()


if __name__ == "__main__":

    min_n = 100
    max_n = 10000
    n_measures = 10

    description = (
        "This test evaluates the time complexity of the evolution\n"
        "process using the big_O package. The measurement is made\n"
        "with respect to the population size and the number of generations\n"
        f"to compute, taken from {min_n} to {max_n}, with {n_measures}\n"
        "equally distributed samples. This may take several minutes.\n"
    )

    print(description)
    best_const_gen, _ = big_o.big_o(
        evolve_const_gen,
        big_o.datagen.n_,
        min_n=min_n,
        max_n=max_n,
        n_measures=n_measures,
    )

    logger.info(f"Time complexity as function of the population size: {best_const_gen}")

    best_const_pop, _ = big_o.big_o(
        evolve_const_pop,
        big_o.datagen.n_,
        min_n=min_n,
        max_n=max_n,
        n_measures=n_measures,
    )

    logger.info(
        f"Time complexity as function of the number of generations: {best_const_pop}"
    )
