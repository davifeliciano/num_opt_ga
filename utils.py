from numpy.typing import ArrayLike, NDArray
import numpy as np

number = float | int


def number_to_bin(value: number, bits: int) -> NDArray:
    """
    Convert the closest integer of a value to its binary
    representation, stored in a np.ndarray.
    """

    value = int(np.rint(value))
    bits = abs(bits)
    max_value = 2**bits - 1
    error_msg = f"Could not convert {value}. {bits} can only represent integers up to {max_value}"

    if value > max_value:
        raise ValueError(error_msg)

    bits_str = np.binary_repr(value, width=bits)
    bits_list = [int(bit) for bit in bits_str]
    return np.array(bits_list, dtype=int)


def bin_to_number(array: ArrayLike) -> NDArray:
    """
    Given a matrix, if not in binary form, convert it putting 0 where 0,
    1 otherwise and convert each of its columns to the respective
    base 10 integer and return the result as a vector
    """

    array = np.array(array, dtype=int).transpose()
    binary_array = (array != 0).astype(int)
    return binary_array.dot(2 ** np.arange(binary_array.shape[-1])[::-1])
