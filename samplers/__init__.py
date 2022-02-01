"""
Support functions for sampling from nonstandard distributions using MC methods.
Currently support rejection sampling, but will add additional methods as they are needed.
"""

from ._rejection_sampler import ProposalDistribution, rejection_sampler
