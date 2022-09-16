from __future__ import annotations
from typing import Callable, List, Sequence
from numpy.typing import ArrayLike, NDArray
import numpy as np
from numpy.random import default_rng


def is_multiple(value: int, number: int) -> bool:
    return value % number == 0


def next_multiple_of(value: int, number: int) -> int:
    return value + number - value % number


class Region:
    """
    Represents a rectangular subregion of R^n, given a 2-column array-like
    Each row represents a range in the respective dimension
    """

    def __init__(self, lims: ArrayLike) -> None:

        lims_array = np.array(lims, dtype=float).transpose()

        if lims_array.ndim != 2 or lims_array.shape[0] != 2:
            raise ValueError("The argument for Region() must be a 2-row array-like")

        self._lims = np.sort(lims_array, axis=0)

    def __repr__(self) -> str:
        return f"Region(lims={self.lims.tolist()})"

    @property
    def lims(self) -> NDArray:
        return self._lims

    @property
    def min(self) -> NDArray:
        return self.lims[0]

    @property
    def max(self) -> NDArray:
        return self.lims[1]

    @property
    def dim(self) -> int:
        return self.lims.shape[1]

    def contains(self, pos: ArrayLike) -> bool:
        """
        Check if the given position vector is contained by the search region
        """

        pos_array = np.array(pos)

        if pos_array.ndim != 1 or pos_array.size != self.dim:
            raise ValueError(
                "The argument of contains() must be a vector with the "
                "same dimension of the region"
            )

        greater_than_min = np.greater_equal(pos_array, self.min).all()
        less_than_max = np.less_equal(pos_array, self.max).all()

        if greater_than_min and less_than_max:
            return True

        return False


class NumericalOptimizationGA:
    """
    A Genetic Algorithm for Numerical Optimization
    """

    VALID_RECOMB_TYPES = ("one_point", "two_point", "random")

    def __init__(
        self,
        search_region: ArrayLike,
        function: Callable[[NDArray], float],
        bits: int = 32,
        pop_size: int = 100,
        fit_func_param: float = 2.0,
        elite: Sequence[int] = (6, 6, 10),
        mut_probs: Sequence[float] = (0.05, 0.05),
        recomb_type: str = "two_point",
    ) -> None:

        # Creating the rng to be used
        self._rng = default_rng()

        self._gen = 0
        self._search_region = Region(search_region)
        self._function = function
        self._bits = int(bits)
        self._fit_func_param = abs(float(fit_func_param))

        # Creating a matrix of conversion that will be used to
        # convert binary arrays into a vector of (Naturals)^(self.bits)
        self._conversion_array = np.array(
            [[2**i for i in range(self.bits)] for _ in range(self.dim)], dtype=int
        ).transpose()

        # Checking pop_size : it should be multiple of 4
        pop_size = abs(int(pop_size))
        if not is_multiple(pop_size, 4) or pop_size == 0:
            pop_size = next_multiple_of(pop_size, 4)
            print(
                f"Argument pop_size of {self.__class__.__name__} need "
                f"to be a non-null multiple of 4. Using pop_size={pop_size}"
            )

        # Checking elite : should be a sequence of 3 ints, all greater than 2,
        #                  their sum must not exceed half the pop_size,
        #                  elite[0] and elite[1] must be even and multiples of 3,
        #                  elite[0] + elite[1] < pop_size // 2,
        #                  elite[2] < (pop_size - elite[0] - elite[1]) // 2
        for value in elite:
            if value < 2:
                raise ValueError("The values in elite must be all greater than 2")

        try:
            if len(elite) < 3:
                raise ValueError("Invalid elite lenght.")
            for value in elite[:2]:
                if not isinstance(value, int):
                    raise ValueError("Invalid type in elite values")
        except (TypeError, ValueError):
            print("The value of elite must be a sequence of 3 ints")
            raise

        elites = []
        for i, value in enumerate(elite[:2]):
            if is_multiple(value, 2) and is_multiple(value, 3):
                elites.append(int(value))
            else:
                print("The first value of elite must be a multiple of both 2 and 3")
                next_multiple_of_3 = next_multiple_of(value, 3)
                next_valid = next_multiple_of_3 + 3 * (next_multiple_of_3 % 2)
                print(f"Using elite[{i}]={next_valid} instead")
                elites.append(next_valid)

        if sum(elite) > pop_size // 2:
            raise ValueError(
                "The sum of the values of elite must not "
                "excced half of the size of the population"
            )

        if elite[0] + elite[1] > pop_size // 2:
            raise ValueError(
                "The sum of the first two values of elite must not "
                "excced half of the size of the population"
            )

        if elite[2] > (pop_size - elite[0] - elite[1]) // 2:
            raise ValueError(
                "The third value in elite must not exceed half the "
                "size of the non-elite population"
            )

        elites.append(elite[2])
        self._elite = tuple(elites)

        # Checking mut_probs : it must be a sequence of 2 floats
        try:
            if len(elite) < 2:
                raise ValueError("Invalid mut_prob size")
            for value in mut_probs[:1]:
                if not isinstance(value, float):
                    raise ValueError("Invalid type in mut_prob values")
        except (TypeError, ValueError):
            print("The value of mut_prob must be a sequence of 3 floats")
            raise

        # Checking recomb_type
        if recomb_type not in self.VALID_RECOMB_TYPES:
            print(f"Valid recomb_types are {self.VALID_RECOMB_TYPES}")
            print(f"recomb_type={recomb_type} was given. Using 'two_point' instead")
            recomb_type = "two_point"

        self._mut_probs = tuple(mut_probs)
        self._recomb_type = recomb_type

        # Creating slices objects to iterate over elite in the recomb process
        self._elite_zero_slices, self._elite_one_slices = self._get_elite_slices()

        # Creating 2 halfs of the population with uniformly distributed individuals
        self._population = [
            self._rng.choice((0, 1), size=(pop_size // 2, self.bits, self.dim))
            for _ in range(2)
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
    def fit_func_param(self) -> float:
        return self._fit_func_param

    @property
    def population(self) -> List[NDArray]:
        return self._population

    @property
    def pop_size(self) -> int:
        return 2 * self.population[0].shape[0]

    @property
    def elite(self) -> Sequence[int]:
        return self._elite

    @property
    def mut_probs(self) -> Sequence[float]:
        return self._mut_probs

    @property
    def recomb_type(self) -> str:
        return self._recomb_type

    def _get_elite_slices(self):
        """
        Return a 2-tuple of 3-tuple of slices objects to use
        later in the iteration over the elite of the population
        """

        elite_slices = []
        for elite, start in zip(self.elite[:2], (0, self.elite[0])):
            elite_third = elite // 3
            elite_slice = (
                slice(start + elite_third),
                slice(start + elite_third, 2 * elite_third),
                slice(start + 2 * elite_third, self.elite[0]),
            )
            elite_slices.append(elite_slice)

        return tuple(elite_slices)

    def _decimal_to_pos(self, array: NDArray) -> NDArray:
        """
        Convert a vector of (Naturals)^(self.dim) to a
        vector of the search region
        """
        search_region_range = self.search_region.max - self.search_region.min
        return (
            self.search_region.min
            + (search_region_range / (2**self.bits - 1)) * array
        )

    def get_pos_array(self, array: NDArray) -> NDArray:
        """
        Convert a population NDArray to a matrix
        of vectors in the search space
        """
        decimal_array = np.sum(self._conversion_array * array, axis=1)
        return np.apply_along_axis(
            func1d=self._decimal_to_pos,
            axis=1,
            arr=decimal_array,
        )

    def fit_function(self, func_values: NDArray) -> NDArray:
        """
        Evaluation of a linear fitness function, given an array-like
        of values for the objective function
        """

        min_func_value = func_values.min()
        max_func_value = func_values.max()
        h = self.fit_func_param
        mu = func_values.mean()

        if min_func_value >= (h * mu - min_func_value) / (h - 1):
            a = mu * (h - 1) / (max_func_value - mu)
            b = mu * (max_func_value - h * mu) / (max_func_value - mu)
        else:
            a = mu / (mu - min_func_value)
            b = -mu * min_func_value / (mu - min_func_value)

        return a * func_values + b

    def selection(self, repeat: bool = False) -> None:
        """
        Perform selection based on the value of the attribute tuple, and the
        selection probabilities evaluated by means of the fitness function
        """

        best_individual_list = []
        selected_individuals_list = []
        # Half the population size
        pop_size = self.pop_size // 2
        # Number of individuals to constitute the elite of the population
        elite_size = sum(self.elite[:2])
        # Number of individuals to select
        roulette_size = pop_size - elite_size

        for array in self.population:
            pos_array = self.get_pos_array(array)
            func_values = np.apply_along_axis(self.function, axis=1, arr=pos_array)
            fit_func_values = self.fit_function(func_values)
            selection_probs = np.clip(
                fit_func_values / fit_func_values.sum(),
                a_min=0.0,
                a_max=1.0,
            )
            best_individual_index = np.argmax(selection_probs)
            best_individual = array[best_individual_index]
            best_individual_list.append(best_individual)
            selected_indexes = self._rng.choice(
                np.arange(pop_size, dtype=int),
                size=roulette_size,
                replace=repeat,
                p=selection_probs,
            )
            # Random positions to insert the best individual among the selected ones
            positions = self._rng.choice(
                np.arange(elite_size, roulette_size, dtype=int),
                size=self.elite[2],
                replace=False,
            )
            selected_individuals = np.take(array, axis=0, indices=selected_indexes)
            selected_individuals[positions] = best_individual
            selected_individuals_list.append(selected_individuals)

        for array, best_individual, selected_individuals in zip(
            self._population, best_individual_list, selected_individuals_list
        ):
            array[self._elite_zero_slices[2]] = best_individual
            array[self._elite_one_slices[2]] = best_individual
            for index in (0, 1):
                array[self._elite_zero_slices[index]] = best_individual_list[index]
                array[self._elite_one_slices[index]] = best_individual_list[index]
            array[elite_size:] = selected_individuals

    def recombination(self) -> None:
        """
        Perform recombination over neighboring pairs of the selected
        population and add their offsprings to the population
        """
        half_pop_size = self.pop_size // 2

        # Create a recombination array-mask
        recomb_array = self._rng.choice(
            (0, 1), size=(half_pop_size, self.bits, self.dim)
        )

        not_recomb_array = np.logical_not(recomb_array)
        elite_size = sum(self.elite[:2])
        roulette_size = half_pop_size - elite_size
        mut_prob_arrays = [
            np.full(shape=(size, self.bits, self.dim), fill_value=mut_prob)
            for size, mut_prob in zip((elite_size, roulette_size), self.mut_probs)
        ]
        mut_prob_array = np.concatenate(mut_prob_arrays, axis=0)

        # Create an array-mask that indicates which genes will suffer a mutation
        random_array = self._rng.random(size=(half_pop_size, self.bits, self.dim))
        mutation_array = np.less(random_array, mut_prob_array)

        # Generate new populations
        first_new_population = np.logical_xor(
            np.logical_or(
                np.logical_and(self.population[0], recomb_array),
                np.logical_and(self.population[1], not_recomb_array),
            ),
            mutation_array,
        ).astype(int)

        second_new_population = np.logical_xor(
            np.logical_or(
                np.logical_and(self.population[0], not_recomb_array),
                np.logical_and(self.population[1], recomb_array),
            ),
            mutation_array,
        ).astype(int)

        self._population = [first_new_population, second_new_population]

    def evolve(self, repeat: bool = False) -> None:
        """
        Perform selection and recombination, and produces the next
        generation of individuals
        """

        self.selection(repeat=repeat)
        self.recombination()
        self._gen += 1

    def sorted_positions(self, reverse: bool = False) -> NDArray:
        """
        Return the population list sorted by the fit_function
        """
        populations = np.concatenate(self.population, axis=0)
        pos_array = self.get_pos_array(populations)
        func_values = np.apply_along_axis(self.function, axis=1, arr=pos_array)
        fit_func_values = self.fit_function(func_values).tolist()
        indexes = [i for i in range(self.pop_size)]
        sorted_indexes = sorted(
            indexes, key=lambda index: fit_func_values[index], reverse=not reverse
        )
        return np.take(pos_array, indices=sorted_indexes, axis=0)

    def best(self, count: int = 1) -> NDArray:
        """
        Return a list with the n best individuals in the population
        """

        if count == 1:
            populations = np.concatenate(self.population, axis=0)
            pos_array = self.get_pos_array(populations)
            func_values = np.apply_along_axis(self.function, axis=1, arr=pos_array)
            return pos_array[np.argmax(func_values)]
        return self.sorted_positions()[:count]
