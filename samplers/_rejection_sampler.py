import types
from typing import Union

from numpy import empty, ndarray
from scipy.stats import norm, uniform, rv_continuous

__all__ = ["ProposalDistribution", "rejection_sampler"]


class ProposalDistribution:
    """
    Wrapper to unify scipy.stats distributions with custom distribution functions.
    """

    def __init__(self, sampler=None, density=None):
        self.sample = sampler
        self.density = density

    def from_scipy(self, scipy_distribution, *args, **kwargs):
        """Provides an alternate constructor to easily instantiate ProposalDistribution from a scipy distribution."""
        self.sample = scipy_distribution(*args, **kwargs).rvs
        self.density = scipy_distribution(*args, **kwargs).pdf
        return self

    def sample(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def density(self, *args, **kwargs):
        return self.density(*args, **kwargs)


def rejection_sampler(
    f: types.FunctionType,
    g: Union[types.FunctionType, types.MethodType, rv_continuous],
    M: float,
    N: int,
) -> ndarray:
    """
    Draw N samples from a density f using rejection sampling.

    inputs:
    -------
    f: function
        A bounded function to sample from using rejection sampling.
        The function does not need to be normalized (to integrate to 1), but it must be bounded
        so that f(x) <= Mg(x) for all x in support(f).

    g: class ProposalDistribution or scipy.stats.rv_continuous
        A class with a sampling attribute and a density function to evaluate.
        g is used as a proposal distribution in rejection sampling, so it must be sampled from and
        its density computed at sampled points x.
        The ProposalDistribution class can be used to pass custom density functions into the sampler.
        When a scipy distribution is used, it will be wrapped in

    M: float
        Some constant such that f(x) <= M * g(x) for all x in support(g).

    N: integer
        The total number of samples to draw.
    """
    if not isinstance(f, types.FunctionType):
        raise TypeError("f must be a function")

    if not isinstance(g, Union[ProposalDistribution, rv_continuous]):
        raise TypeError(
            "g must be a continuous scipy distribution or a .ProposalDistribution"
        )

    if isinstance(g, rv_continuous):
        g = ProposalDistribution().from_scipy(g)

    if not isinstance(M, Union[int, float]):
        raise TypeError("M must be an integer or float to scale samples from g by")

    if M <= 0:
        raise ValueError("M must be positive")

    if not isinstance(N, int):
        raise TypeError("N must be an integer")

    if N <= 0:
        raise ValueError("N must be positive")

    # Create storage to put accepted samples in
    samples = empty(N)
    # Need a number to track the total number of samples drawn at each step in the algorithm
    n = 0
    # Will run in a while loop, so need an exit condition to avoid infinite loops
    # Track the total number of iterations. If it exceeds 2 * M iterations, then exit with
    # the total number of samples drawn and a warning that we exited early
    k = 0

    def proposal(x):
        """Helper to get values of f(x) and f(x) / M * g(x)"""
        return  f(x) / (M * g.density(x))

    # Now ready to implement rejection sampling
    # f is the target density, g is the proposal distribution
    # draw x ~ g, u ~ U(0, 1) and compare u ~ f(x) / (M * g(x))
    while n < N:
        # Increment k to preserve an exit condition
        if k > M * M * N:
            break
        k += 1

        x = g.sample()
        u = uniform().rvs()

        # Rejection sampling: compare a uniform random variable with f(x) / (M * g(x))
        # If u <= f(x) / (M * g(x)), accept x as a sample from f
        if u <= proposal(x):
            samples[n] = x
            n += 1

    return samples


def main():
    f = lambda x: 2 * norm().pdf(x) * norm().cdf(x)
    g = ProposalDistribution().from_scipy(norm)
    M = 2
    N = 1000

    samples = rejection_sampler(f, g, M, N)

    return samples


if __name__ == "__main__":
    main()
