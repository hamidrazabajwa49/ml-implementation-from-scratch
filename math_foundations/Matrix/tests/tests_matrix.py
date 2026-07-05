"""
test_matrix.py
===============

Comprehensive pytest suite for ``matrix.py``.

Run with::

    pytest test_matrix.py -v

Requires: pytest, numpy (for regression/cross-checking only -- the
library itself has no NumPy dependency). Assumes ``vector.py`` (from
Module 1) is importable from the same directory.
"""

import os
import sys
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matrix import DimensionMismatchError, Matrix, NotSquareError, SingularMatrixError
from Vectors.vector import Vector

ATOL = 1e-4


def to_np(m: Matrix) -> np.ndarray:
    return np.array([row.components for row in m.rows], dtype=float)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
class TestConstruction:
    def test_basic_construction(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.shape == (2, 2)

    def test_empty_matrix(self):
        m = Matrix([])
        assert m.shape == (0, 0)

    def test_single_empty_row(self):
        m = Matrix([[]])
        assert m.shape == (1, 0)

    def test_ragged_rows_raise(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3]])

    def test_non_numeric_component_raises(self):
        with pytest.raises(TypeError):
            Matrix([[1, "a"], [2, 3]])

    def test_bool_component_rejected(self):
        with pytest.raises(TypeError):
            Matrix([[1, True], [2, 3]])

    def test_repr_empty(self):
        assert repr(Matrix([])) == "Matrix([])"

    def test_is_square(self):
        assert Matrix([[1, 2], [3, 4]]).is_square is True
        assert Matrix([[1, 2, 3]]).is_square is False
        assert Matrix([]).is_square is False

    def test_copy_is_independent(self):
        m = Matrix([[1, 2], [3, 4]])
        c = m.copy()
        c.rows[0].components[0] = 999
        assert m.rows[0].components[0] == 1


# ----------------------------------------------------------------------
# Container protocol / equality
# ----------------------------------------------------------------------
class TestContainerProtocol:
    def test_getitem_row(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m[0].components == [1, 2]

    def test_iteration(self):
        m = Matrix([[1, 2], [3, 4]])
        assert [row.components for row in m] == [[1, 2], [3, 4]]

    def test_len(self):
        assert len(Matrix([[1, 2], [3, 4]])) == 2

    def test_equality(self):
        assert Matrix([[1, 2]]) == Matrix([[1, 2]])
        assert Matrix([[1, 2]]) != Matrix([[1, 3]])

    def test_equality_different_shape(self):
        assert (Matrix([[1, 2]]) == Matrix([[1, 2], [3, 4]])) is False

    def test_equality_with_non_matrix(self):
        assert (Matrix([[1, 2]]) == "x") is False

    def test_unhashable(self):
        with pytest.raises(TypeError):
            hash(Matrix([[1, 2]]))


# ----------------------------------------------------------------------
# Arithmetic vs NumPy
# ----------------------------------------------------------------------
class TestArithmetic:
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    npA = np.array([[1, 2], [3, 4]], dtype=float)
    npB = np.array([[5, 6], [7, 8]], dtype=float)

    def test_add_matrices(self):
        np.testing.assert_allclose(to_np(self.A + self.B), self.npA + self.npB)

    def test_add_scalar_broadcasts(self):
        np.testing.assert_allclose(to_np(self.A + 10), self.npA + 10)
        np.testing.assert_allclose(to_np(10 + self.A), self.npA + 10)

    def test_sub_matrices(self):
        np.testing.assert_allclose(to_np(self.A - self.B), self.npA - self.npB)

    def test_rsub_scalar(self):
        np.testing.assert_allclose(to_np(10 - self.A), 10 - self.npA)

    def test_add_shape_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            self.A + Matrix([[1, 2, 3]])

    def test_matmul_matrices(self):
        np.testing.assert_allclose(to_np(self.A * self.B), self.npA @ self.npB)
        np.testing.assert_allclose(to_np(self.A @ self.B), self.npA @ self.npB)

    def test_matmul_dimension_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            Matrix([[1, 2]]) * Matrix([[1, 2]])

    def test_scalar_mul(self):
        np.testing.assert_allclose(to_np(self.A * 2), self.npA * 2)
        np.testing.assert_allclose(to_np(2 * self.A), self.npA * 2)

    def test_matvec(self):
        v = Vector([1, 1])
        result = self.A.matvec(v)
        np.testing.assert_allclose(result.components, self.npA @ np.array([1, 1]))

    def test_mul_by_vector_dispatches_to_matvec(self):
        v = Vector([1, 1])
        result = self.A * v
        assert isinstance(result, Vector)

    def test_matvec_dimension_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            self.A.matvec(Vector([1, 2, 3]))

    def test_truediv_scalar(self):
        np.testing.assert_allclose(to_np(self.A / 2), self.npA / 2)

    def test_truediv_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            self.A / 0

    def test_mul_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            self.A * "x"

    def test_neg(self):
        np.testing.assert_allclose(to_np(-self.A), -self.npA)

    def test_matrix_power_positive(self):
        m = Matrix([[1, 1], [0, 1]])
        np.testing.assert_allclose(to_np(m ** 3), np.linalg.matrix_power(np.array([[1, 1], [0, 1]], dtype=float), 3))

    def test_matrix_power_negative_uses_inverse(self):
        m = Matrix([[1, 1], [0, 1]])
        np.testing.assert_allclose(to_np(m ** -1), np.linalg.inv(np.array([[1, 1], [0, 1]], dtype=float)), atol=1e-6)

    def test_matrix_power_non_int_raises(self):
        with pytest.raises(TypeError):
            Matrix([[1, 2], [3, 4]]) ** 1.5


# ----------------------------------------------------------------------
# Element-wise helpers
# ----------------------------------------------------------------------
class TestElementWise:
    def test_element_wise(self):
        m = Matrix([[1, 2], [3, 4]]).element_wise(lambda x: x ** 2)
        assert m == Matrix([[1, 4], [9, 16]])

    def test_element_wise_with(self):
        m = Matrix([[1, 2]]).element_wise_with(Matrix([[3, 4]]), lambda a, b: a + b)
        assert m == Matrix([[4, 6]])

    def test_element_wise_with_shape_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            Matrix([[1, 2]]).element_wise_with(Matrix([[1, 2, 3]]), lambda a, b: a + b)


# ----------------------------------------------------------------------
# Structural utilities
# ----------------------------------------------------------------------
class TestStructural:
    def test_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(to_np(m.transpose()), np.array([[1, 4], [2, 5], [3, 6]], dtype=float))

    def test_transpose_empty(self):
        assert Matrix([]).transpose().shape == (0, 0)

    def test_columns(self):
        m = Matrix([[1, 2], [3, 4]])
        cols = m.columns()
        assert cols[0].components == [1, 3]
        assert cols[1].components == [2, 4]

    def test_diagonal(self):
        m = Matrix([[1, 2], [3, 4]])
        d = m.diagonal()
        assert d == Matrix([[1, 0], [0, 4]])

    def test_is_symmetric(self):
        assert Matrix([[1, 2], [2, 3]]).is_symmetric() is True
        assert Matrix([[1, 2], [3, 4]]).is_symmetric() is False

    def test_frobenius_norm(self):
        m = Matrix([[3, 4]])
        assert m.frobenius_norm() == pytest.approx(5.0)

    def test_zeros(self):
        assert Matrix.zeros(2, 3) == Matrix([[0, 0, 0], [0, 0, 0]])

    def test_zeros_negative_raises(self):
        with pytest.raises(ValueError):
            Matrix.zeros(-1, 2)

    def test_identity(self):
        assert Matrix.identity(2) == Matrix([[1, 0], [0, 1]])

    def test_identity_non_positive_raises(self):
        with pytest.raises(ValueError):
            Matrix.identity(0)


# ----------------------------------------------------------------------
# Gaussian elimination: REF, rank, determinant, inverse
# ----------------------------------------------------------------------
class TestElimination:
    def test_rank_full(self):
        assert Matrix([[1, 0], [0, 1]]).rank() == 2

    def test_rank_deficient(self):
        assert Matrix([[1, 2], [2, 4]]).rank() == 1

    def test_rank_zero_matrix(self):
        assert Matrix([[0, 0], [0, 0]]).rank() == 0

    def test_rank_empty(self):
        assert Matrix([]).rank() == 0

    @pytest.mark.parametrize(
        "data",
        [
            [[1, 2], [3, 4]],
            [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 10]],
            [[1, 2], [2, 4]],  # singular
        ],
    )
    def test_determinant_matches_numpy(self, data):
        m = Matrix(data)
        npm = np.array(data, dtype=float)
        assert m.determinant() == pytest.approx(np.linalg.det(npm), abs=1e-6)

    def test_determinant_non_square_raises(self):
        with pytest.raises(NotSquareError):
            Matrix([[1, 2, 3], [4, 5, 6]]).determinant()

    def test_determinant_empty_raises(self):
        with pytest.raises(NotSquareError):
            Matrix([]).determinant()

    def test_determinant_1x1(self):
        assert Matrix([[7]]).determinant() == pytest.approx(7.0)

    @pytest.mark.parametrize(
        "data",
        [[[4, 7], [2, 6]], [[1, 2, 3], [0, 1, 4], [5, 6, 0]]],
    )
    def test_inverse_matches_numpy(self, data):
        m = Matrix(data)
        npm = np.array(data, dtype=float)
        inv = m.inverse()
        np.testing.assert_allclose(to_np(inv), np.linalg.inv(npm), atol=1e-6)

    def test_inverse_of_singular_raises(self):
        with pytest.raises(SingularMatrixError):
            Matrix([[1, 2], [2, 4]]).inverse()

    def test_inverse_non_square_raises(self):
        with pytest.raises(NotSquareError):
            Matrix([[1, 2, 3], [4, 5, 6]]).inverse()

    def test_inverse_times_original_is_identity(self):
        m = Matrix([[4, 7], [2, 6]])
        product = m * m.inverse()
        assert product == Matrix([[1, 0], [0, 1]], tol=1e-6)


# ----------------------------------------------------------------------
# Trace / characteristic polynomial
# ----------------------------------------------------------------------
class TestTraceAndCharPoly:
    def test_trace(self):
        assert Matrix([[1, 2], [3, 4]]).trace() == 5

    def test_trace_non_square_raises(self):
        with pytest.raises(NotSquareError):
            Matrix([[1, 2, 3]]).trace()

    @pytest.mark.parametrize(
        "data",
        [
            [[2, 0], [0, 3]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 10]],
            [[2, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]],
        ],
    )
    def test_characteristic_poly_matches_numpy(self, data):
        m = Matrix(data)
        npm = np.array(data, dtype=float)
        np.testing.assert_allclose(m.characteristic_poly(), np.poly(npm), atol=1e-4)

    def test_characteristic_poly_consistent_with_determinant(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        n = m.n_rows
        coeffs = m.characteristic_poly()
        assert m.determinant() == pytest.approx(((-1) ** n) * coeffs[-1], abs=1e-6)


# ----------------------------------------------------------------------
# QR decomposition
# ----------------------------------------------------------------------
class TestQR:
    def test_qr_reconstructs_original(self):
        data = [[1, 2], [3, 4], [5, 6]]
        m = Matrix(data)
        Q, R = m.qr_decompose()
        np.testing.assert_allclose(to_np(Q * R), np.array(data, dtype=float), atol=1e-6)

    def test_qr_columns_orthonormal(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        Q, _ = m.qr_decompose()
        cols = Q.columns()
        for i, ci in enumerate(cols):
            assert ci.norm() == pytest.approx(1.0, abs=1e-6)
            for cj in cols[i + 1 :]:
                assert abs(ci.dot(cj)) < 1e-6


# ----------------------------------------------------------------------
# Eigenvalues / eigenvectors
# ----------------------------------------------------------------------
class TestEigen:
    def test_eigenvalues_1x1(self):
        assert Matrix([[5]]).eigenvalues() == [5.0]

    def test_eigenvalues_2x2_symmetric_matches_numpy(self):
        data = [[2, 1], [1, 2]]
        m = Matrix(data)
        evals = sorted(m.eigenvalues())
        np_evals = sorted(np.linalg.eigvals(np.array(data, dtype=float)).tolist())
        np.testing.assert_allclose(evals, np_evals, atol=1e-6)

    def test_eigenvalues_3x3_symmetric_matches_numpy(self):
        data = [[4, 1, 1], [1, 3, 0], [1, 0, 2]]
        m = Matrix(data)
        evals = sorted(m.eigenvalues())
        np_evals = sorted(np.linalg.eigvals(np.array(data, dtype=float)).tolist())
        np.testing.assert_allclose(evals, np_evals, atol=1e-4)

    def test_eigenvalues_complex_conjugate_pair_matches_numpy(self):
        """Regression test for the QR-algorithm bug: a pure rotation matrix
        has genuinely complex eigenvalues, which must not be silently
        truncated to (wrong) real diagonal entries."""
        theta = 0.7
        data = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        m = Matrix(data)
        evals = m.eigenvalues()
        assert any(isinstance(e, complex) and abs(e.imag) > 1e-6 for e in evals)
        np_evals = np.linalg.eigvals(np.array(data, dtype=float))
        evals_sorted = sorted(evals, key=lambda z: (z.imag if isinstance(z, complex) else 0.0))
        np_sorted = sorted(np_evals.tolist(), key=lambda z: z.imag)
        for a, b in zip(evals_sorted, np_sorted):
            assert abs(complex(a) - b) < 1e-6

    def test_eigenvalues_non_square_raises(self):
        with pytest.raises(NotSquareError):
            Matrix([[1, 2, 3], [4, 5, 6]]).eigenvalues()

    def test_eigenvectors_satisfy_av_eq_lambda_v(self):
        data = [[4, 1, 1], [1, 3, 0], [1, 0, 2]]
        m = Matrix(data)
        for lam in m.eigenvalues():
            for v in m.eigenvectors(lam):
                Av = m.matvec(v)
                lv = v * lam
                for a, b in zip(Av.components, lv.components):
                    assert a == pytest.approx(b, abs=1e-4)

    def test_eigenvectors_of_complex_eigenvalue_raises(self):
        theta = 0.7
        data = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        m = Matrix(data)
        complex_eval = next(e for e in m.eigenvalues() if isinstance(e, complex))
        with pytest.raises(NotImplementedError):
            m.eigenvectors(complex_eval)

    def test_eigenvectors_large_magnitude_matrix_not_empty(self):
        """Regression test: A^T A style large-magnitude matrices must still
        find eigenvectors for closely-spaced small eigenvalues (scale-aware
        tolerance fix)."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        G = m.transpose() * m
        for lam in G.eigenvalues():
            assert len(G.eigenvectors(lam)) >= 1

    def test_power_iteration_dominant_eigenvalue(self):
        m = Matrix([[2, 0], [0, 5]])
        lam, v = m.power_iteration(seed=42)
        assert lam == pytest.approx(5.0, abs=1e-4)

    def test_power_iteration_reproducible_with_seed(self):
        m = Matrix([[2, 1], [1, 2]])
        lam1, _ = m.power_iteration(seed=7)
        lam2, _ = m.power_iteration(seed=7)
        assert lam1 == lam2

    def test_power_iteration_zero_matrix_raises(self):
        with pytest.raises(ValueError):
            Matrix([[0, 0], [0, 0]]).power_iteration()

    def test_power_iteration_non_square_raises(self):
        with pytest.raises(NotSquareError):
            Matrix([[1, 2, 3]]).power_iteration()

    def test_diagonalize_reconstructs_original(self):
        data = [[4, 1], [2, 3]]
        m = Matrix(data)
        P, D = m.diagonalize()
        recon = P * D * P.inverse()
        np.testing.assert_allclose(to_np(recon), np.array(data, dtype=float), atol=1e-4)

    def test_spectral_theorem_symmetric(self):
        m = Matrix([[2, 1], [1, 2]])
        result = m.spectral_theorem()
        assert result == {
            "symmetric": True,
            "real_eigenvalues": True,
            "orthogonal_eigenvectors": True,
        }

    def test_spectral_theorem_non_symmetric(self):
        result = Matrix([[1, 2], [3, 4]]).spectral_theorem()
        assert result["symmetric"] is False


# ----------------------------------------------------------------------
# SVD
# ----------------------------------------------------------------------
class TestSVD:
    @pytest.mark.parametrize(
        "data",
        [
            [[1, 2], [3, 4], [5, 6]],  # tall
            [[1, 2, 3], [4, 5, 6]],  # wide
            [[4, 0], [3, -5]],  # square
            [[1, 2, 3], [4, 5, 6], [7, 8, 10]],  # square, non-symmetric
        ],
    )
    def test_svd_reconstructs_original(self, data):
        m = Matrix(data)
        npm = np.array(data, dtype=float)
        U, Sigma, Vt = m.svd()
        k = min(m.shape)
        assert U.shape == (m.n_rows, k)
        assert Sigma.shape == (k, k)
        assert Vt.shape == (k, m.n_cols)
        recon = Matrix.reconstruct(U, Sigma, Vt)
        np.testing.assert_allclose(to_np(recon), npm, atol=ATOL)

    def test_singular_values_match_numpy(self):
        data = [[1, 2], [3, 4], [5, 6]]
        m = Matrix(data)
        npm = np.array(data, dtype=float)
        _, Sigma, _ = m.svd()
        mine = sorted((Sigma.rows[i].components[i] for i in range(Sigma.n_rows)), reverse=True)
        theirs = sorted(np.linalg.svd(npm, compute_uv=False).tolist(), reverse=True)
        np.testing.assert_allclose(mine, theirs, atol=1e-4)

    def test_low_rank_approx_full_rank_is_exact(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        m = Matrix(data)
        approx = m.low_rank_approx(3)
        np.testing.assert_allclose(to_np(approx), np.array(data, dtype=float), atol=ATOL)

    def test_low_rank_approx_matches_numpy_truncated_svd(self):
        data = [[1, 2], [3, 4], [5, 6]]
        m = Matrix(data)
        npm = np.array(data, dtype=float)
        approx1 = m.low_rank_approx(1)
        U, S, Vt = np.linalg.svd(npm, full_matrices=False)
        S1 = S.copy()
        S1[1:] = 0
        np_approx1 = U @ np.diag(S1) @ Vt
        np.testing.assert_allclose(to_np(approx1), np_approx1, atol=ATOL)

    def test_low_rank_approx_invalid_k_raises(self):
        m = Matrix([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            m.low_rank_approx(5)
        with pytest.raises(ValueError):
            m.low_rank_approx(-1)

    def test_compression_ratio_full_rank_is_no_savings_or_negative(self):
        m = Matrix([[1, 2], [3, 4]])
        stats = m.compression_ratio(2)
        assert stats["original_elements"] == 4

    def test_compression_ratio_invalid_k_raises(self):
        m = Matrix([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            m.compression_ratio(10)


# ----------------------------------------------------------------------
# NaN / Inf handling
# ----------------------------------------------------------------------
class TestNanInf:
    def test_pure_arithmetic_propagates_nan(self):
        m = Matrix([[1, float("nan")]]) + Matrix([[1, 1]])
        assert math.isnan(m.rows[0].components[1])

    def test_determinant_with_nan_raises_clear_error(self):
        """Regression test: NaN comparisons are always False, so naive
        pivot-selection would silently skip NaN and return a wrong (but
        finite-looking) determinant instead of failing loudly."""
        m = Matrix([[1, float("nan")], [2, 3]])
        with pytest.raises(ValueError, match="finite"):
            m.determinant()

    def test_inverse_with_inf_raises_clear_error(self):
        m = Matrix([[1, float("inf")], [2, 3]])
        with pytest.raises(ValueError, match="finite"):
            m.inverse()

    def test_eigenvalues_with_nan_raises_clear_error(self):
        m = Matrix([[1, float("nan")], [2, 3]])
        with pytest.raises(ValueError, match="finite"):
            m.eigenvalues()

    def test_row_echelon_form_with_nan_raises(self):
        m = Matrix([[float("nan"), 1], [2, 3]])
        with pytest.raises(ValueError, match="finite"):
            m.row_echelon_form()


# ----------------------------------------------------------------------
# Custom exception hierarchy
# ----------------------------------------------------------------------
class TestExceptionHierarchy:
    def test_dimension_mismatch_is_value_error(self):
        assert issubclass(DimensionMismatchError, ValueError)

    def test_singular_matrix_is_value_error(self):
        assert issubclass(SingularMatrixError, ValueError)

    def test_not_square_is_value_error(self):
        assert issubclass(NotSquareError, ValueError)
