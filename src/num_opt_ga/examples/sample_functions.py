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
from dataclasses import dataclass, field
from itertools import count
from ..num_opt_ga import NumericalOptimizationGA

# Setting up argparse
description = (
    "A tool to visualize the Numerical Optimization Genetic Algorithm "
    "in action. It evolves a number of populations that match the number "
    "of logical processors in the machine, with the given number of "
    "individuals, for the specified number of generations. Being E1 and E2 integers "
    "supplied by the user with option -e, the process is done for three sample functions, "
    "and after it, two plots are rendered: one with the position of the best E1 "
    "individuals of each population along with the level curves of each function, "
    "and another with the evolution of the function values for the best E2 "
    "individuals along the generations"
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
    nargs=2,
    default=[3, 3],
    metavar=("E1", "E2"),
    help=(
        "two integers E1 and E2. Two plots will be rendered: one with the position of "
        "the best E1 individuals of each population along with the level curves of "
        "each function, and another with the evolution of the function values for "
        "the best E2 individuals along the generations."
    ),
)

args = parser.parse_args()

# Setting up logger
LOG_FORMAT = "%(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger("logger")
logger.setLevel("INFO")


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
    sigmas = (0.3, 0.03)
    amps = (0.8, 0.88)
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


plot_types = ("position", "func_evolution")
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
elite = args.elite

# Looking for ValueErrors
for i, value in enumerate(elite):
    if value > pop_size:
        logger.warning(
            f"The value of e{i + 1} must be less or equal to the size "
            f"of the populations. Using e{i + 1}=3 instead"
        )
        elite[i] = 3


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


@dataclass
class ProcessData:
    """
    Contains a NumericalOptimizationGA instance and a list
    of lists to append function values of elite individuas for
    each generation
    """

    ga: NumericalOptimizationGA
    gens_history: List = field(default_factory=lambda: [0])
    func_values_history_list: List = field(init=False)

    def __post_init__(self):
        self.func_values_history_list = [
            [self.ga.function(best.pos)] for best in self.ga.best(elite[1])
        ]

    def evolve(self) -> None:
        self.ga.evolve()
        self.gens_history.append(self.ga.gen)
        for individual, func_value_history in zip(
            self.ga.best(elite[1]), self.func_values_history_list
        ):
            func_value_history.append(self.ga.function(individual.pos))


def evolve_ga(process_data: ProcessData) -> ProcessData | None:
    try:
        while process_data.ga.gen < gens:
            process_data.evolve()
        return process_data
    # Ignore KeyboardInterrupt on a child process
    except KeyboardInterrupt:
        return None


@timer
def evolve_gas(
    process_data_list: List[ProcessData], processes: int
) -> List[ProcessData]:
    with mp.Pool(processes) as pool:
        try:
            return pool.map(evolve_ga, process_data_list)
        except KeyboardInterrupt:
            # Kill the pool when KeyboardInterrupt is raised
            logger.info(f"Process terminated by the user")
            pool.terminate()
            pool.join()
            sys.exit(1)


@timer
def save_fig(fig, filename: str, dpi: int) -> None:
    fig.savefig(filename, dpi=dpi)


# Color generator to use in time evolution plots
def colors():
    for i in count():
        yield f"C{i}"


if __name__ == "__main__":

    # Creating a function at top level since the process pool demands
    # NumericalOptimizationGA to be fully pickable
    def ga_function(pos: NDArray) -> float:
        return function(*pos)

    for func_name, function in functions.items():
        logger.info(f"Optmizing function {func_name}")
        logger.info(f"Initializing {pop_count} populations with {pop_size} individuals")

        process_data_list = [
            ProcessData(
                ga=NumericalOptimizationGA(
                    search_region=((-1, 1), (-1, 1)),
                    function=ga_function,
                    bits=32,
                    pop_size=pop_size,
                )
            )
            for _ in range(pop_count)
        ]

        logger.info(f"Evolving populations for {gens} generations")
        process_data_list = evolve_gas(process_data_list, processes=pop_count)

        # Figure with the position of the best E1 (elite[0]) individuals
        logger.info(
            f"Creating figure with the position of the best {elite[0]} individuals of each population"
        )

        fig_3d = plt.figure(figsize=[10.8, 4.8])
        ax_contour = fig_3d.add_subplot(1, 2, 1)
        ax_3d = fig_3d.add_subplot(1, 2, 2, projection="3d")

        # Setting up axis labels
        ax_contour.set(xlabel="x", ylabel="y")
        ax_3d.set(xlabel="x", ylabel="y", zlabel="f(x, y)")

        x = y = np.linspace(-1, 1, 250)
        x, y = np.meshgrid(x, y)

        ax_contour.contour(x, y, function(x, y), levels=10, cmap=cm.jet, zorder=-1)

        elite_pop = []
        for process_data in process_data_list:
            elite_pop += process_data.ga.best(elite[0])

        # Ploting figure
        for individual in elite_pop:
            ax_contour.scatter(*individual.pos, marker="+", color="black", lw=1.0)

        ax_3d.plot_trisurf(
            x.flatten(),
            y.flatten(),
            function(x, y).flatten(),
            cmap=cm.jet,
        )

        logger.info("Done!")
        filename = f"position_{func_name}.png"
        logger.info(f"Saving Figure as {filename}")
        save_fig(fig_3d, filename, dpi=300)

        # Figure with the time evolution of the best E2 (elite[1]) individuals
        logger.info(
            f"Creating figure with the time evolution of the best {elite[1]} "
            "individuals of each population over the generations"
        )
        fig_time_evol, ax_time_evol = plt.subplots()

        # Setting up axis labels
        ax_time_evol.set(xlabel="gen", ylabel="f(x, y)")
        # Number of populations to include in the figure
        fig_pop_count = 4 if pop_count > 3 else 2
        for process_data, color in zip(process_data_list[:fig_pop_count], colors()):
            for func_values_history in process_data.func_values_history_list:
                ax_time_evol.plot(
                    process_data.gens_history,
                    func_values_history,
                    color=color,
                    linewidth=0.5,
                    alpha=0.2,
                )

        logger.info("Done!")
        filename = f"evolution_{func_name}.png"
        logger.info(f"Saving Figure as {filename}")
        save_fig(fig_time_evol, filename, dpi=500)

        print(os.get_terminal_size().columns * "-")
