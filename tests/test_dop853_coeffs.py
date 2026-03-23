import numpy as np
import pytest

from gnc_toolkit.integrators.dop853_coeffs import A, C, B, E3, E5

def test_dop853_array_sizes():
    # Verify coefficients have correct lengths and dimensions
    assert C.shape == (16,)
    assert A.shape == (16, 16)
    # B is defined as a slice: A[12, :12], so B has 12 elements
    assert B.shape == (12,)
    assert E3.shape == (13,)
    assert E5.shape == (13,)

def test_dop853_sum_rules():
    # For Dormand-Prince method, sum of row elements in A should roughly equal C[i]
    # Check rows up to 12
    for i in range(12):
        row_sum = np.sum(A[i, :])
        assert np.isclose(row_sum, C[i], atol=1e-10)
