import sys
import os
import logging
import time
import multiprocessing as mp
from typing import List
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from num_opt_ga import NumericalOptimizationGA

# Setting up logger
LOG_FORMAT = "%(asctime)s : %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger("logger")
logger.setLevel("INFO")

# Setting up Matplotlib
plt.rcParams.update(
    {
        "figure.figsize": [10.8, 4.8],
    }
)


def function_1(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    n = 9
    sigma = 0.4
    return np.cos(n * np.pi * r) * np.exp(-((r / sigma) ** 2))


def function_2(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    xs = (0.5, 0.6)
    ys = (0.5, 0.1)
    rs = [np.sqrt((x - xc) ** 2 + (y - yc) ** 2) for xc, yc in zip(xs, ys)]
    sigmas = (0.3, 0.1)
    amps = (0.8, 1.0)
    exps = [
        amp * np.exp(-((r / sigma) ** 2)) for amp, r, sigma in zip(amps, rs, sigmas)
    ]
    return sum(exps)


def function_3(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    xs = (0.5, 0.1, -0.2, -0.3)
    ys = (0.5, -0.6, -0.3, 0.4)
    rs = [np.sqrt((x - xc) ** 2 + (y - yc) ** 2) for xc, yc in zip(xs, ys)]
    sigmas = (0.4, 0.3, 0.5, 0.2)
    amps = (-0.5, 0.7, -0.3, 0.4)
    exps = [
        amp * np.exp(-((r / sigma) ** 2)) for amp, r, sigma in zip(amps, rs, sigmas)
    ]
    return sum(exps)


def to_string(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return (
            f"{minutes} {'minute' if minutes == 1 else 'minutes'} and "
            f"{seconds} {'second' if seconds == 1 else 'seconds'}"
        )
    return f"{seconds:.2f} {'second' if seconds == 1 else 'seconds'}"


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"Done in {to_string(elapsed)}")
        return result

    return wrapper


def evolve_ga(ga: NumericalOptimizationGA) -> NumericalOptimizationGA | None:
    try:
        while ga.gen < gens:
            ga.evolve()
        return ga
    # Ignore KeyboardInterrupt on a child process
    except KeyboardInterrupt:
        return None


@timer
def evolve_gas(
    ga_list: List[NumericalOptimizationGA], processes: int
) -> List[NumericalOptimizationGA]:
    with mp.Pool(processes) as pool:
        try:
            return pool.map(evolve_ga, ga_list)
        except KeyboardInterrupt:
            # Kill the pool when KeyboardInterrupt is raised
            logger.info(f"Process terminated by the user")
            pool.terminate()
            pool.join()
            sys.exit(1)


@timer
def save_fig(fig, filename: str) -> None:
    fig.savefig(filename, dpi=200)


functions = (function_1, function_2, function_3)
pop_count = mp.cpu_count()
pop_size = 100
gens = 200
best_count = 3
terminal_cols = os.get_terminal_size().columns

if __name__ == "__main__":

    # Creating a function at top level since the process pool demands
    # NumericalOptimizationGA to be fully pickable
    def ga_function(pos: NDArray) -> float:
        return function(*pos)

    for i, function in enumerate(functions):
        logger.info(f"Optmizing Sample Function {i + 1}")
        logger.info(f"Initializing {pop_count} populations with {pop_size} individuals")

        gas = [
            NumericalOptimizationGA(
                search_region=((-1, 1), (-1, 1)),
                function=ga_function,
                bits=32,
                pop_size=pop_size,
            )
            for _ in range(pop_count)
        ]

        logger.info(f"Evolving populations for {gens} generations")
        gas = evolve_gas(gas, processes=pop_count)

        logger.info(
            f"Creating figure with the {best_count} individuals of each population"
        )
        fig = plt.figure()
        ax_contour = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

        # Setting up axis labels
        ax_contour.set(xlabel="x", ylabel="y")
        ax_3d.set(xlabel="x", ylabel="y", zlabel="f(x, y)")

        x = y = np.linspace(-1, 1, 250)
        x, y = np.meshgrid(x, y)

        ax_contour.contour(x, y, function(x, y), levels=10, cmap=cm.jet, zorder=-1)

        elite_pop = []
        for ga in gas:
            elite_pop += ga.best(best_count)

        for individual in elite_pop:
            ax_contour.scatter(*individual.pos, marker="+", color="black", lw=1.0)

        ax_3d.plot_trisurf(
            x.flatten(),
            y.flatten(),
            function(x, y).flatten(),
            cmap=cm.jet,
        )

        logger.info("Done!")
        filename = f"sample_function_{i + 1}.png"
        logger.info(f"Saving Figure as {filename}")
        save_fig(fig, filename)
        print(terminal_cols * "-")
