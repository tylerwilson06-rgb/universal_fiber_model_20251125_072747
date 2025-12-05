"""
Test Fixtures Package

Contains signal generators and test data for the Universal Fiber Sensor Model test suite.
"""

from .signals import (
    generate_synthetic_signal,
    generate_das_like_signal,
    generate_multichannel_signal,
    generate_edge_case_signals,
)

__all__ = [
    'generate_synthetic_signal',
    'generate_das_like_signal',
    'generate_multichannel_signal',
    'generate_edge_case_signals',
]

