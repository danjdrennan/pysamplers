import pytest
import warnings

from numpy import ndarray
from scipy.stats import norm, uniform

from src.rejection_sampler import ProposalDistribution, rejection_sampler


"""
Test suite for hw1 code.
"""

class TestProposalDistribution:
    """
    Tests for instantiation of ProposalDistribution.
    ProposalDistribution unifies sampling and computing densities for custom functions with scipy's rv_continuous class.
    The test must cover instantiation using two constructors, the default (__init__ method) and a helper (from_scipy method).
    """
    def test_init_constructor(self):
        g = ProposalDistribution(norm().rvs, norm().pdf)
        assert isinstance(g, ProposalDistribution)
        assert isinstance(g.sample(5), ndarray)
        assert g.density(0) == norm().pdf(0)
    
    def test_scipy_constructor(self):
        g = ProposalDistribution().from_scipy(norm)
        assert isinstance(g, ProposalDistribution)
    
class TestRejectionSampler:
    """Tests for rejection_sampler function"""
    g = ProposalDistribution(norm().rvs, norm().pdf)
    
    def test_input_f(self):
        """Check type errors on input f"""

        with pytest.raises(TypeError):
            rejection_sampler(1, self.g, 1, 1)
            rejection_sampler("f", self.g, 1, 1)
            rejection_sampler([1], self.g, 1, 1)
    
    def test_input_g(self):
        """Check type errors raised for inputs on M"""
        f = lambda x: x
        with pytest.raises(TypeError):
            rejection_sampler(f, 1, 1, 1)
            rejection_sampler(f, "g", 1, 1)
            rejection_sampler(f, [1], 1, 1)
            rejection_sampler(f, self.f, 1, 1)
        
    def test_input_M(self):
        """Check type errors raised for inputs on M"""
        f = lambda x: x
        with pytest.raises(ValueError):
            rejection_sampler(f, self.g, -1, 1)
            rejection_sampler(f, self.g, 0, 1)

        with pytest.raises(TypeError):
            rejection_sampler(f, self.g, [5], 1)
            rejection_sampler(f, self.g, '5', 1)
        
    def test_input_N(self):
        """Test for input errors on arg N"""
        f = lambda x: x
        with pytest.raises(ValueError):
            rejection_sampler(f, self.g, 5, -1)
            rejection_sampler(f, self.g, 5, 0)
        
        with pytest.raises(TypeError):
            rejection_sampler(f, self.g, 1, '0')
            rejection_sampler(f, self.g, 1, [1])

    def test_sampling(self):
        
        assert isinstance(
            rejection_sampler(lambda x: 1, self.g, 1, 5),
            ndarray
        )
