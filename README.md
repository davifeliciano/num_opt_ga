# num_opt_ga

A Python implementation of a genetic algorithm for numerical optimization
of real functions

## Usage

The algorithm is encapsulated in the `NumericalOptimizationGA` object,
which consists basically of a list of `Individual` objects and some methods
to interact with it. Suppose you want to optimize the function of $\mathbb{R}^2$
in $\mathbb{R}$

$$ f(x,y) = \cos(9\pi r)\exp\left(- \frac{r^2}{0.4^2}\right) $$

$r$ being the distance of $(x, y)$ from the origin. Then we must pass a rectangular
subregion of $\mathbb{R}^2$ as a sequence of 2 tuples, each one representing an interval on
the $x$ and $y$ axes respectively. The function itself must be passed to the object
constructor. Its call signature must be such that it will be called with a
vector of $\mathbb{R}^2$ (as an one dimensional NumPy NDArray) and return a float. We can
also specify the amount of individuals that will constitute the population.

```python
from num_opt_ga import NumericalOptimizationGA
import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm


def func_to_optimize(pos: NDArray) -> float:
    r = norm(pos)
    sigma = 0.4
    return np.cos(9 * np.pi * r) * np.exp(-((r / sigma) ** 2))


ga = NumericalOptimizationGA(
        search_region=((-1, 1), (-1, 1)),
        function=func_to_optimize,
        pop_size=100,
     )
```

Now the object `ga` contains a list `ga.population` of `Individual` objects.
Each in,dividual is on a random position of the given search space. This
position is stored on the `pos` atribute as an NumPy one dimensional NDArray.
The next generation of individuals is computed by a call to the `evolve` method,
which will replace all the individuals in the list. So we can evolve the
population for 100 generation with

```python
for _ in range(100):
    ga.evolve()
```

Now we can create a scatter plot with the position of the best individual

```python
best_pos = [individual.pos for individual in ga.best(5)]

import matplotlib.pyplot as plt

best_pos = [individual.pos for individual in ga.best()]
best_x = [pos[0] for pos in best_pos]
best_y = [pos[1] for pos in best_pos]

fig, ax = plt.subplots()
ax.set(
    xlabel="x",
    ylabel="y",
    xlim=(-1, 1),
    ylim=(-1, 1),
)
ax.scatter(best_x, best_y, color="red")
plt.show()
```

yielding

![Result](https://user-images.githubusercontent.com/26972046/168857024-625dab54-59b9-42e3-aebc-5289fe28e511.png)

as expected, since the global maximum of this particular function is located at
$r = 0$. For other examples of usage, check the [examples
submodule](https://github.com/davifeliciano/num_opt_ga/tree/main/src/num_opt_ga/examples).
