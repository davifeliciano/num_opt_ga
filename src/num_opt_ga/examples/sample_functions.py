import sys
import os
import argparse
import logging
import time
import multiprocessing as mp
from typing import List
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ..num_opt_ga import NumericalOptimizationGA

# Setting up argparse
description = (
    "A tool to visualize the Numerical Optimization Genetic Algorithm "
    "in action. It evolves a number of populations that match the number "
    "of logical processors in the machine, with the given number of "
    "individuals, for the specified number of generations. Being e an integer "
    "supplied by the user, that process is done for three sample functions, "
    "and after it, the position of the best e individuals of each population "
    "are ploted as a black cross along with the level curves of each function."
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "-f",
    "--functions",
    type=str,
    choices=("all", "damped_cossine", "near_gaussians", "sparse_gaussians"),
    nargs="*",
    default=["all"],
    help="the functions to optimize",
)

parser.add_argument(
    "-p",
    "--pop-size",
    type=int,
    nargs="?",
    default=100,
    const=100,
    help="the number of individuals to compose the populations.",
)

parser.add_argument(
    "-g",
    "--gens",
    type=int,
    nargs="?",
    default=300,
    const=300,
    help=(
        "the number of generations to evaluate in the evolution "
        "process of each population."
    ),
)

parser.add_argument(
    "-e",
    "--elite",
    type=int,
    nargs="?",
    default=3,
    const=3,
    help=(
        "the number of individuals to compose the elite of each "
        "population. The best e individual will have its position "
        "ploted in the result."
    ),
)

args = parser.parse_args()

# Setting up logger
LOG_FORMAT = "%(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger("logger")
logger.setLevel("INFO")

# Setting up Matplotlib
plt.rcParams.update(
    {
        "figure.figsize": [10.8, 4.8],
    }
)


def distance(
    x: NDArray | float,
    y: NDArray | float,
    xc: NDArray | float,
    yc: NDArray | float,
) -> NDArray | float:
    """
    Computes the distance between (x, y) and (xc, yc)
    """

    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def gaussian(
    amp: NDArray | float,
    r: NDArray | float,
    sigma: NDArray | float,
) -> NDArray | float:
    """
    A Gaussian in polar coordinates with amplitude amp and width sigma
    """

    return amp * np.exp(-((r / sigma) ** 2))


def damped_cossine(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    r = distance(x, y, 0.5, 0.5)
    return np.cos(9 * np.pi * r) * gaussian(1.0, r, 0.4)


def near_gaussians(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    xs = (0.5, 0.6)
    ys = (0.5, 0.1)
    rs = [distance(x, y, xc, yc) for xc, yc in zip(xs, ys)]
    sigmas = (0.3, 0.1)
    amps = (0.8, 1.0)
    gaussians = [gaussian(amp, r, sigma) for amp, r, sigma in zip(amps, rs, sigmas)]
    return sum(gaussians)


def sparse_gaussians(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    xs = (0.5, 0.1, -0.2, -0.3)
    ys = (0.5, -0.6, -0.3, 0.4)
    rs = [distance(x, y, xc, yc) for xc, yc in zip(xs, ys)]
    sigmas = (0.4, 0.3, 0.5, 0.2)
    amps = (-0.5, 0.7, -0.3, 0.4)
    gaussians = [gaussian(amp, r, sigma) for amp, r, sigma in zip(amps, rs, sigmas)]
    return sum(gaussians)


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


functions = {
    "damped_cossine": damped_cossine,
    "near_gaussians": near_gaussians,
    "sparse_gaussians": sparse_gaussians,
}

if "all" not in args.functions:
    functions = {key: functions[key] for key in args.functions}

pop_count = mp.cpu_count()
pop_size = abs(args.pop_size)
gens = abs(args.gens)
elite = abs(args.elite)

# Looking for ValueErrors
if elite > pop_size:
    logger.warning(
        "The number of individuals in the elite must be less or "
        "equal to the size of the populations. Using elite=3 instead"
    )
    elite = 3

if __name__ == "__main__":

    # Creating a function at top level since the process pool demands
    # NumericalOptimizationGA to be fully pickable
    def ga_function(pos: NDArray) -> float:
        return function(*pos)

    for func_name, function in functions.items():
        logger.info(f"Optmizing function {func_name}")
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

        logger.info(f"Creating figure with the {elite} individuals of each population")
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
            elite_pop += ga.best(elite)

        for individual in elite_pop:
            ax_contour.scatter(*individual.pos, marker="+", color="black", lw=1.0)

        ax_3d.plot_trisurf(
            x.flatten(),
            y.flatten(),
            function(x, y).flatten(),
            cmap=cm.jet,
        )

        logger.info("Done!")
        filename = f"{func_name}.png"
        logger.info(f"Saving Figure as {filename}")
        save_fig(fig, filename)
        print(os.get_terminal_size().columns * "-")
