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
    help="the number of individuals to compose the populations. Default is 100.",
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
        "process of each population. Default is 300."
    ),
)

parser.add_argument(
    "-e",
    "--elite",
    type=float,
    nargs=2,
    default=[1.0, 0.2],
    metavar=("E1", "E2"),
    help=(
        "two floats E1 and E2 between 0 and 1. Two plots will be rendered: "
        "one with the position of the best E1 percent of individuals of each "
        "population along with the level curves of each function, and another "
        "with the evolution of the function values for the best E2 percent of "
        "individuals along the generations. Defaults are 1.0 and 0.2, respectively."
    ),
)

parser.add_argument(
    "-m",
    "--mut-probs",
    type=float,
    nargs=2,
    default=[0.1, 0.1],
    metavar=("M1", "M2"),
    help=(
        "two floats M1 and M2 between 0 and 1. M1 is the probability of "
        "mutation on the crossover between members of the elite and M2 is "
        "the probability of mutation of the rest of the population. "
        "Defaults are 0.1 and 0.1, respectively."
    ),
)

parser.add_argument(
    "-t",
    "--tex",
    action="store_true",
    help=(
        "use latex to render the figure labels. When using this option "
        "make sure to have a LaTeX distribution installed, as "
        "well as the mathpazo font package."
    ),
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


# Functions to optimize
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
    logger.info(f"Saving figure as {filename}")
    fig.savefig(filename, dpi=dpi)


def is_percentage(value: float) -> bool:
    return 0 <= value <= 1


# Color generator to use in time evolution plots
def colors():
    for i in count():
        yield f"C{i}"


if __name__ == "__main__":

    # Parsing cl arguments
    args = parser.parse_args()
    defaults = parser.parse_args(args=[])

    # Setting up logger
    LOG_FORMAT = "%(levelname)s : %(asctime)s : %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logger = logging.getLogger("logger")
    logger.setLevel("INFO")

    plot_types = ("position", "func_evolution")
    functions = {
        "damped_cossine": damped_cossine,
        "near_gaussians": near_gaussians,
        "sparse_gaussians": sparse_gaussians,
    }

    # Checking cl arguments
    if "all" not in args.functions:
        functions = {key: functions[key] for key in args.functions}

    pop_count = mp.cpu_count()
    pop_size = abs(args.pop_size)
    gens = abs(args.gens)
    elite_probs = args.elite
    mut_probs = args.mut_probs

    for i, (elite_value, mut_prob_value) in enumerate(zip(elite_probs, mut_probs)):
        if not is_percentage(elite_value):
            logger.warning(
                f"The value of E{i + 1} must be between 0 and 1. "
                f"Using E{i + 1} = {defaults.elite[i]} instead."
            )
            elite_probs[i] = defaults.elite[i]
        if not is_percentage(mut_prob_value):
            logger.warning(
                f"The value of M{i + 1} must be between 0 and 1. "
                f"Using M{i + 1} = {defaults.mut_probs[i]} instead."
            )
            mut_probs[i] = defaults.mut_probs[i]

    elite = [int(elite_prob * pop_size) for elite_prob in elite_probs]

    elite_points = pop_count * elite[0]
    if elite_points > 10000:
        logger.warning(
            f"The contour plot will have the position of {elite_points} "
            "individuals. The rendering of the image could take a long time."
        )

    elite_curves = pop_count * elite[1]
    if elite_curves > 1800:
        logger.warning(
            f"The evolution plot will have {elite_curves} time series. "
            "You may find dificult to distinguish each population. "
            "Try smaller values of M2."
        )

    if args.tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage[sc]{mathpazo}",
            }
        )

    # Creating a function at top level since the process pool demands
    # NumericalOptimizationGA to be fully pickable
    def ga_function(pos: NDArray) -> float:
        return function(*pos)

    for i, (func_name, function) in enumerate(functions.items()):
        logger.info(f"Optmizing function {func_name}")
        logger.info(f"Initializing {pop_count} populations with {pop_size} individuals")

        process_data_list = [
            ProcessData(
                ga=NumericalOptimizationGA(
                    search_region=((-1, 1), (-1, 1)),
                    function=ga_function,
                    bits=32,
                    pop_size=pop_size,
                    mut_probs=mut_probs,
                )
            )
            for _ in range(pop_count)
        ]

        logger.info(f"Evolving populations for {gens} generations")
        process_data_list = evolve_gas(process_data_list, processes=pop_count)

        # Meshgrid to use in the plots
        x = y = np.linspace(-1, 1, 500)
        x, y = np.meshgrid(x, y)

        # Figure with the graph of the function
        logger.info(f"Creating figure with the graph of the function")
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(projection="3d")

        # Setting up axis and creating the plot
        ax_3d.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=rf"$f_{i + 1}(x, y)$")
        ax_3d.plot_trisurf(
            x.flatten(),
            y.flatten(),
            function(x, y).flatten(),
            cmap=cm.viridis,
        )

        # Saving figure
        filename = f"graph_{func_name}.png"
        save_fig(fig_3d, filename, dpi=300)

        # Figure with the position of the best E1 (elite[0]) individuals
        logger.info(
            "Creating contour plot with the position of the best "
            f"{elite[0]} individuals of each population"
        )
        fig_contour, ax_contour = plt.subplots()

        # Setting up axis and creating the plot
        ax_contour.set(aspect="equal", xlabel=r"$x$", ylabel=r"$y$")
        ax_contour.contour(x, y, function(x, y), levels=10, cmap=cm.viridis, zorder=0)

        # Creating list of elite to scatter on the plot
        elite_pop = []
        best_pop = []
        for process_data in process_data_list:
            elite_pop += process_data.ga.best(elite[0])
            best_pop += process_data.ga.best()

        for individual in elite_pop:
            ax_contour.scatter(
                *individual.pos,
                color="black",
                s=2.0,
                alpha=0.15,
                label="Individuals",
                zorder=-1,
            )

        for individual in best_pop:
            ax_contour.scatter(
                *individual.pos,
                marker="x",
                color="red",
                s=100.0,
                label="Maximums",
                zorder=1,
            )

        # Adding legend
        handles, labels = ax_contour.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_contour.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            framealpha=1.0,
        )

        filename = f"contour_{func_name}.png"
        save_fig(fig_contour, filename, dpi=300)

        # Figure with the time evolution of the best E2 (elite[1]) individuals
        logger.info(
            f"Creating figure with the time evolution of the best {elite[1]} "
            "individuals of each population over the generations"
        )
        fig_time_evol, ax_time_evol = plt.subplots()

        # Setting up axis and creating the plot
        ax_time_evol.set(
            # yscale="log",
            xlabel=r"$g$",
            ylabel=rf"$f_{i + 1}(x, y)$",
        )

        mut_probs_zoom_lim = {
            "damped_cossine": 0.15,
            "near_gaussians": 0.3,
            "sparse_gaussians": 0.0,
        }

        zoom_condition = (
            mut_probs[0] < mut_probs_zoom_lim[func_name]
            and mut_probs[1] < mut_probs_zoom_lim[func_name]
        )

        if zoom_condition:
            ax_zoom = ax_time_evol.inset_axes((0.3, 0.075, 0.65, 0.55))
            zoom_xlim = 4 * gens // 10
            zoom_ylims = {
                "damped_cossine": (0.7, 1.05),
                "near_gaussians": (0.75, 1.05),
            }
            ax_zoom.set(
                xlim=(-1, zoom_xlim),
                ylim=zoom_ylims[func_name],
                # xticklabels=[],
                yticklabels=[],
            )
            ax_time_evol.indicate_inset_zoom(ax_zoom, edgecolor="black", alpha=1.0)

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
                if zoom_condition:
                    ax_zoom.plot(
                        process_data.gens_history[: zoom_xlim + 1],
                        func_values_history[: zoom_xlim + 1],
                        color=color,
                        linewidth=0.5,
                        alpha=0.2,
                    )

        filename = f"evolution_{func_name}.png"
        save_fig(fig_time_evol, filename, dpi=500)
        print(os.get_terminal_size().columns * "-")
