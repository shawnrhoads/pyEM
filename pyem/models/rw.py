"""
Rescorla-Wagner model aliases for backward compatibility.
"""
from .rl import rw1a1b_simulate as rw_simulate, rw1a1b_fit as rw_fit

__all__ = ['rw_simulate', 'rw_fit']