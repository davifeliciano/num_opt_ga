from __future__ import annotations
from typing import List, Optional
from numpy.typing import ArrayLike, NDArray
from numpy.random import default_rng
import numpy as np
from utils import number_to_bin, bin_to_number

rng = default_rng()


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


class Individual:
    """
    Represents an individual of a population, with its characteristics
    encoded by a given number of chromosomes with a given number of bits
    """

    def __init__(
        self,
        bits: int,
        search_region: Region,
        init_pos: Optional[ArrayLike] = None,
        init_dna: Optional[ArrayLike] = None,
    ) -> None:

        self._bits = int(bits)

        if isinstance(search_region, Region):
            self._search_region = search_region
        else:
            try:
                self._search_region = Region(search_region)
            except ValueError:
                print(
                    "Could not interpret the given search_region as a subregion of R^n"
                )
                raise

        # Evaluate the dna based on init_pos or init_dna
        if init_pos is None:
            self._init_pos = None
            if init_dna is None:
                self._init_dna = None
                min_lim = self._search_region.min
                max_lim = self._search_region.max
                dim = self._search_region.dim
                pos = (max_lim - min_lim) * rng.random(size=dim) + min_lim
                self._dna = self.dna_from_pos(pos)
            else:
                dna = np.array(init_dna)
                self._check_dna_argument(dna)
                self._init_dna = dna
                self._dna = dna
        else:
            pos = np.array(init_pos)
            self._check_pos_argument(pos)
            self._init_pos = pos
            self._dna = self.dna_from_pos(pos)

        # Evaluate the position based on the dna
        self._pos = self.pos_from_dna()

    def __repr__(self) -> str:
        lims_list = self.search_region.lims.transpose().tolist()
        init_arg = ")"

        if self.init_pos is not None:
            init_arg = f", init_pos={self.init_pos.tolist()})"
        elif self.init_dna is not None:
            init_arg = f", init_dna={self.init_dna.tolist()})"

        return f"Individual(bits={self.bits}, search_region={lims_list}" + init_arg

    @property
    def bits(self) -> int:
        return self._bits

    @property
    def search_region(self) -> Region:
        return self._search_region

    @property
    def dim(self) -> int:
        return self.search_region.dim

    @property
    def init_dna(self) -> None | NDArray:
        return self._init_dna

    @property
    def init_pos(self) -> None | NDArray:
        return self._init_pos

    @property
    def dna(self) -> NDArray:
        return self._dna

    @property
    def pos(self) -> NDArray:
        return self._pos

    @classmethod
    def new_from(cls, instance: Individual) -> Individual:
        """
        Create an individual of the same species of the instance
        passed (same search_region and same dna shape)
        """

        return cls(
            instance.bits, instance.search_region, instance.init_pos, instance.init_dna
        )

    def _check_dna_argument(self, dna: NDArray) -> None:
        """
        Raises ValueError if the dna argument is not valid
        """

        expected_shape = (self.bits, self.search_region.dim)
        if dna.shape != expected_shape:
            raise ValueError(f"Any dna array-like must have the shape {expected_shape}")

    def _check_pos_argument(self, pos: NDArray) -> None:
        """
        Raises ValueError if the pos argument is not valid
        """

        if pos.ndim != 1 or pos.size != self.search_region.dim:
            raise ValueError(
                "Any position vector must have the same dimension of the search_region"
            )

        if not self.search_region.contains(pos):
            raise ValueError(
                "Any position vector must be contained by the given search_region"
            )

    def dna_from_pos(self, pos: ArrayLike) -> NDArray:
        """
        Given the position of the individual in the search space,
        return the correspondent dna encoded as a matrix (np.ndarray)
        """

        pos_array = np.array(pos)
        self._check_pos_argument(pos_array)
        min_lim = self.search_region.min
        max_lim = self.search_region.max
        numbers = (pos_array - min_lim) * (2**self.bits - 1) / (max_lim - min_lim)
        chromos = [number_to_bin(number, self.bits) for number in numbers]
        return np.array(chromos, dtype=int).transpose()

    def pos_from_dna(self, dna: Optional[ArrayLike] = None) -> NDArray:
        """
        Given some matrix which encodes some individual dna,
        return its correspondent position in the search space
        as a np.ndarray
        """

        if dna is None:
            dna_array = self.dna
        else:
            dna_array = np.array(dna, dtype=int)
            self._check_dna_argument(dna_array)

        min_lim = self.search_region.min
        max_lim = self.search_region.max
        numbers = bin_to_number(dna_array)
        return numbers * (max_lim - min_lim) / (2**self.bits - 1) + min_lim

    def same_species(self, other: Individual) -> bool:
        """
        Check if the given individual is of the same species
        by comparing the shape of the dna matrix and the
        reference of its search_region
        """

        dna_shape_check = self.dna.shape == other.dna.shape
        search_region_check = self.search_region is other.search_region
        return dna_shape_check and search_region_check

    def recombine_with(
        self,
        other: Individual,
        recomb_type: str = "two_point",
        mut_prob: float | ArrayLike = 0.05,
    ) -> List[Individual]:
        """
        Perform recombination with other individual if they
        are both of the same species and return two offsprings
        as a list, after mutation
        """

        if not self.same_species(other):
            raise ValueError("Cannot recombine individuals of different species")

        valid_types = ("one_point", "two_point", "random")
        if recomb_type not in valid_types:
            print(
                "Valid recomb_type values in recombine_with() are 'one_point', "
                "'two_point' and 'random'"
            )
            print(f"recomb_type={recomb_type} was given. Using 'two_point' instead")
            recomb_type = "two_point"

        # Create a recombination array
        if recomb_type == "random":
            recomb_array = rng.choice((0, 1), size=self.dna.shape)
        else:
            recomb_array = np.zeros_like(self.dna)
            if recomb_type == "one_point":
                first_point = rng.integers(0, self.bits, size=self.dim)
                for column, row in enumerate(first_point):
                    recomb_array[row:, column] = 1

            if recomb_type == "two_point":
                first_point = rng.integers(0, self.bits - 2, size=self.dim)
                second_point = rng.integers(first_point + 1, self.bits, size=self.dim)
                for column, row in enumerate(zip(first_point, second_point)):
                    recomb_array[row[0] : row[1], column] = 1

        not_recomb_array = np.logical_not(recomb_array)

        # Create a array with the probabilities of mutation of each gene
        if isinstance(mut_prob, (int, float)):
            if mut_prob < 0.0:
                print("The mut_prob argument in recombine() must be positive")
                print("Picking the absolute value of the supplied value instead")
            if mut_prob > 1.0:
                print("The mut_prob argument in recombine() must be in [0, 1]")
                print("Picking the decimal part of the supplied value instead")

            mut_prob = abs(mut_prob % 1)
            mut_prob_array = np.full(shape=self.dna.shape, fill_value=mut_prob)
        else:
            mut_prob_array = np.array(mut_prob, dtype=float)
            self._check_dna_argument(mut_prob_array)
            array_min = mut_prob_array.min()
            array_max = mut_prob_array.max()
            mut_prob_array = (mut_prob_array - array_min) / (array_max - array_min)

        # Create an array-mask that indicates which genes will suffer a mutation
        random_array = rng.random(size=self.dna.shape)
        mutation_array = np.less(random_array, mut_prob_array)

        # Perform recombination
        dnas = []
        dnas.append(
            np.logical_or(
                np.logical_and(self.dna, recomb_array),
                np.logical_and(other.dna, not_recomb_array),
            )
        )

        dnas.append(
            np.logical_or(
                np.logical_and(self.dna, not_recomb_array),
                np.logical_and(other.dna, recomb_array),
            )
        )

        # Perform mutation by doing exclusive_or(dna, mutation_array)
        dnas = [np.logical_xor(dna, mutation_array).astype(int) for dna in dnas]

        offsprings = [
            Individual(bits=self.bits, search_region=self.search_region, init_dna=dna)
            for dna in dnas
        ]

        # print("Recombination Array =", recomb_array, sep="\n", end="\n\n")
        # print("Mutation Array =", mutation_array, sep="\n", end="\n\n")
        return offsprings
