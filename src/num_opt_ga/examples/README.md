# Examples

This is a submodule containing examples of use of the `num_opt_ga` module. Each
script of this submodule is described below

## sample_functions

A command line tool to visualize the Numerical Optimization Genetic Algorithm in
action. It evolves a number of populations that match the number of logical
processors in the machine, with the given number of individuals, for the
specified number of generations. Being `E1` and `E2` integers supplied by the
user with option `-e E1 E2` or `--elite E1 E2`, the process is done for three
sample functions, and after it, two plots are rendered: one with the position of
the best `E1` individuals of each population along with the level curves of each
function, and another with the evolution of the function values for the best
`E2` individuals along the generations. 

A help message with all the compatible options and the usage is printed with

```bash
python -m num_opt_ga.examples.sample_functions --help
```

Run the example with the default values of all the options with

```bash
python -m num_opt_ga.examples.sample_functions
```

The functions available for testing are

- damped_cossine
- near_gaussians
- sparse_gaussians

all mapping $\left[-1, 1\right] \times \left[-1, 1\right]$ to $\mathbb{R}$.

### damped_cossine

A circular cossine damped by a gaussian

$$ r = \sqrt{\left(x - 0.5\right)^2 + \left(y - 0.5\right)^2} $$

$$ f(x,y) = \cos(9\pi r)\exp\left(- \frac{r^2}{0.4^2}\right) $$

### near_gaussians

Two gaussians near each other

$$ r_1 = \sqrt{\left(x - 0.5\right)^2 + \left(y - 0.5\right)^2} $$

$$ r_2 = \sqrt{\left(x - 0.6\right)^2 + \left(y - 0.1\right)^2} $$

$$ f(x,y) = 0.8 \exp\left(-\frac{r_1^2}{0.3^2}\right) + 0.88 \exp\left(-
\frac{r_2^2}{0.03^2}\right) $$

### sparse_gaussians

Four gaussians scattered in the search space

$$ r_1 = \sqrt{\left(x - 0.5\right)^2 + \left(y - 0.5\right)^2} $$

$$ r_2 = \sqrt{\left(x - 0.1\right)^2 + \left(y + 0.6\right)^2} $$

$$ r_3 = \sqrt{\left(x + 0.2\right)^2 + \left(y - 0.3\right)^2} $$

$$ r_4 = \sqrt{\left(x + 0.3\right)^2 + \left(y + 0.4\right)^2} $$

$$ f(x,y) = - 0.5 \exp\left(-\frac{r_1^2}{0.4^2}\right) + 0.7 \exp\left(- \frac{r_2^2}{0.3^2}\right) -
0.3 \exp\left(- \frac{r_3^2}{0.5^2}\right) + 0.3 \exp\left(- \frac{r_4^2}{0.3^2}\right) $$
