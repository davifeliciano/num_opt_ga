from __future__ import annotations
from typing import Callable, List, Sequence
from numpy.typing import ArrayLike, NDArray
import numpy as np
from individual import rng, Region, Individual
from utils import is_multiple, next_multiple_of


class NumericalOptimizationGA:
    """
    A Genetic Algorithm for Numerical Optimization
    """

    def __init__(
        self,
        search_region: ArrayLike,
        function: Callable[[NDArray], float],
        bits: int = 32,
        pop_size: int = 100,
        fit_func_type: str = "linear",
        fit_func_param: float = 2.0,
        elite: Sequence[int] = (4, 6, 10),
        mut_probs: Sequence[float] = (0.05, 0.05),
    ) -> None:

        self._gen = 0
        self._search_region = Region(search_region)
        self._function = function
        self._bits = int(bits)
        self._fit_func_param = abs(float(fit_func_param))

        # Checking pop_size
        pop_size = int(pop_size)
        if not is_multiple(pop_size, 4):
            pop_size = next_multiple_of(pop_size, 4)
            print("pop_size need to be a multiple of 4")
            print(f"Using pop_size={pop_size}")

        # Checking fit_func_type
        valid_types = ("linear", "boltzmann")
        if fit_func_type not in valid_types:
            print("Valid fit_func_type values are 'linear' and 'boltzmann'")
            print(f"fit_func_type={fit_func_type} was given. Using 'linear' instead")
            self._fit_func_type = "linear"
        else:
            self._fit_func_type = fit_func_type

        if self._fit_func_type == "linear":
            self._fit_function = self.linear_fit_function
        else:
            self._fit_function = self.boltzmann_fit_function

        # Checking elite
        for value in elite:
            if value < 2:
                raise ValueError("The values in elite must be all greater than 2")

        try:
            if len(elite) < 3:
                raise ValueError("Invalid elite size")
            for value in elite[:2]:
                if not isinstance(value, int):
                    raise ValueError("Invalid type in elite values")
        except (TypeError, ValueError):
            print("The value of elite must be a sequence of 3 ints")
            raise

        elites = []
        for value in elite[:2]:
            if is_multiple(value, 2):
                elites.append(int(value))
            else:
                print("The first two values of elite need to be multiples of 2")
                print("Using the nearest greater even values instead")
                elites.append(next_multiple_of(value, 2))

        if sum(elites) > pop_size // 2 // 3:
            raise ValueError(
                "The sum of the first two values of elite must not "
                "excced a sixth of pop_size"
            )

        if elite[2] >= pop_size // 2:
            raise ValueError(
                "The third value in elite must not exceed half the pop_size"
            )

        elites.append(elite[2])
        self._elite = tuple(elites)

        # Checking mut_probs
        try:
            if len(elite) < 2:
                raise ValueError("Invalid mut_prob size")
            for value in mut_probs[:1]:
                if not isinstance(value, float):
                    raise ValueError("Invalid type in mut_prob values")
        except (TypeError, ValueError):
            print("The value of mut_prob must be a sequence of 3 floats")
            raise

        self._mut_probs = tuple(mut_probs)

        # Creating population
        self._population = [
            Individual(self.bits, self.search_region) for _ in range(pop_size)
        ]

    def __repr__(self) -> str:
        lims_list = self.search_region.lims.transpose().tolist()
        return (
            f"NumericalOptimizationGA(search_region={lims_list}, bits={self.bits}, "
            f"fit_func_type={self.fit_func_type}, fit_func_param={self.fit_func_param})"
        )

    @property
    def gen(self) -> int:
        return self._gen

    @property
    def bits(self) -> int:
        return self._bits

    @property
    def dim(self) -> int:
        return self.search_region.dim

    @property
    def search_region(self) -> Region:
        return self._search_region

    @property
    def function(self) -> Callable:
        return self._function

    @property
    def fit_func_type(self) -> str:
        return self._fit_func_type

    @property
    def fit_func_param(self) -> float:
        return self._fit_func_param

    @property
    def fit_function(self) -> Callable:
        return self._fit_function

    @property
    def population(self) -> List[Individual]:
        return self._population

    @property
    def pop_size(self) -> int:
        return len(self.population)

    @property
    def elite(self) -> Sequence[int]:
        return self._elite

    @property
    def mut_probs(self) -> Sequence[float]:
        return self._mut_probs

    @staticmethod
    def _check_fit_func_arg(func_values_array: NDArray) -> None:
        """
        Raises ValueError if its argument does not represent a vector
        """

        if func_values_array.ndim != 1:
            raise ValueError("The argument of linear_fit_func() must be a vector")

    def linear_fit_function(self, func_values: ArrayLike) -> NDArray:
        """
        Evaluation of a linear fitness function, given an array-like
        of values for the objective function
        """

        func_values_array = np.array(func_values)
        self._check_fit_func_arg(func_values_array)
        min_func_value = func_values_array.min()
        max_func_value = func_values_array.max()
        h = self.fit_func_param
        mu = func_values_array.mean()

        if min_func_value >= (h * mu - min_func_value) / (h - 1):
            a = mu * (h - 1) / (max_func_value - mu)
            b = mu * (max_func_value - h * mu) / (max_func_value - mu)
        else:
            a = mu / (mu - min_func_value)
            b = -mu * min_func_value / (mu - min_func_value)

        return a * func_values_array + b

    def boltzmann_fit_function(self, func_values: ArrayLike) -> NDArray:
        """
        A fitness that folows the Boltzmann distribution, evaluated
        over the population
        """

        func_values_array = np.array(func_values)
        self._check_fit_func_arg(func_values_array)
        return np.exp(-self.fit_func_param * func_values_array)

    def selection(self, repeat: bool = False) -> None:
        """
        Perform selection based on the value of the attribute tuple, and the
        selection probabilities evaluated by means of the fitness function
        """

        func_values = [self.function(individual.pos) for individual in self.population]
        fit_func_values = self.fit_function(func_values)
        # Using abs here because for some reason a negative probability
        # (of order e10-18) keeps appearing on the list, maybe due to
        # floating point arithmetic imprecisions
        selection_probs = np.abs(fit_func_values / fit_func_values.sum())
        elite_size = sum(self.elite[:2])
        elite_index = np.argmax(selection_probs)
        roulette_size = self.pop_size // 2 - elite_size
        indexes = rng.choice(
            np.arange(self.pop_size, dtype=int),
            size=roulette_size,
            replace=repeat,
            p=selection_probs,
        )

        elite = [self.population[elite_index] for _ in range(elite_size)]
        roulette = [self.population[index] for index in indexes]

        # Inserting the elite individual in random positions in the roulette
        positions = rng.choice(
            np.arange(roulette_size, dtype=int), size=self.elite[2], replace=False
        )

        for position in positions:
            roulette[position] = self.population[elite_index]

        self._population = elite + roulette

    @staticmethod
    def _offsprings_from_list(
        parents_list: List[Individual], mut_prob: float
    ) -> List[Individual]:
        """
        Given a list of parents, perform recombination over neighboring
        pairs of parents with the given mutation probability and return
        the list of resultant offsprings
        """

        offsprings = []
        for i in range(0, len(parents_list), 2):
            offsprings += parents_list[i].recombine_with(
                parents_list[i + 1], mut_prob=mut_prob
            )
        return offsprings

    def recombination(self) -> None:
        """
        Perform recombination over neighboring pairs of the selected
        population and add their offsprings to the population
        """

        elite = self.population[: self.elite[0]]
        mut_elite = self.population[self.elite[0] : self.elite[1]]
        roulette = self.population[self.elite[1] :]
        offsprings = []
        for parents_list, mut_prob in zip(
            (elite, mut_elite, roulette), (0, *self.mut_probs)
        ):
            offsprings += self._offsprings_from_list(parents_list, mut_prob)

        self._population += offsprings

    def evolve(self, repeat: bool = False) -> None:
        """
        Perform selection and recombination, and produces the next
        generation of individuals
        """

        self.selection(repeat=repeat)
        self.recombination()
        self._gen += 1

    def best(self, count: int = 5) -> List[Individual]:
        """
        Return a list with the n best individuals in the population
        """

        func_values = [self.function(individual.pos) for individual in self.population]
        fit_func_values = self.fit_function(func_values).tolist()
        indexes = [i for i in range(self.pop_size)]
        sorted_indexes = sorted(
            indexes, key=lambda index: fit_func_values[index], reverse=True
        )
        return [self.population[index] for index in sorted_indexes[:count]]
