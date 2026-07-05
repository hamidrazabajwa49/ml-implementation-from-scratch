"""
test_vector.py
===============

Comprehensive pytest suite for ``vector.py``.

Run with::

    pytest test_vector.py -v

Requires: pytest, numpy (for regression/cross-checking only -- the library
itself has no NumPy dependency).
"""

import os
import sys
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector import DimensionMismatchError, Vector, ZeroVectorError

TOL = 1e-9


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
class TestConstruction:
    def test_basic_construction(self):
        v = Vector([1, 2, 3])
        assert v.components == [1, 2, 3]

    def test_empty_vector_is_legal(self):
        v = Vector([])
        assert len(v) == 0

    def test_construction_from_generator(self):
        v = Vector(x for x in range(3))
        assert v.components == [0, 1, 2]

    def test_construction_from_tuple(self):
        v = Vector((1.5, 2.5))
        assert v.components == [1.5, 2.5]

    def test_non_iterable_raises_type_error(self):
        with pytest.raises(TypeError):
            Vector(5)

    def test_string_input_raises_type_error(self):
        # Strings are iterable but each "component" would be a character.
        with pytest.raises(TypeError):
            Vector("123")

    def test_non_numeric_component_raises_type_error(self):
        with pytest.raises(TypeError, match="index 1"):
            Vector([1, "a", 3])

    def test_bool_component_rejected(self):
        with pytest.raises(TypeError):
            Vector([1, True, 3])

    def test_none_component_rejected(self):
        with pytest.raises(TypeError):
            Vector([1, None])

    def test_nan_and_inf_are_accepted_as_valid_floats(self):
        v = Vector([float("nan"), float("inf"), -float("inf")])
        assert math.isnan(v[0])
        assert math.isinf(v[1])


# ----------------------------------------------------------------------
# Container protocol
# ----------------------------------------------------------------------
class TestContainerProtocol:
    def test_len(self):
        assert len(Vector([1, 2, 3])) == 3

    def test_getitem_index(self):
        v = Vector([10, 20, 30])
        assert v[1] == 20

    def test_getitem_slice_returns_vector(self):
        v = Vector([10, 20, 30])
        sliced = v[0:2]
        assert isinstance(sliced, Vector)
        assert sliced.components == [10, 20]

    def test_setitem_valid(self):
        v = Vector([1, 2, 3])
        v[0] = 99
        assert v[0] == 99

    def test_setitem_invalid_type_raises(self):
        v = Vector([1, 2, 3])
        with pytest.raises(TypeError):
            v[0] = "bad"

    def test_setitem_bool_raises(self):
        v = Vector([1, 2, 3])
        with pytest.raises(TypeError):
            v[0] = True

    def test_iteration(self):
        assert list(Vector([1, 2, 3])) == [1, 2, 3]

    def test_equality(self):
        assert Vector([1, 2]) == Vector([1, 2])
        assert Vector([1, 2]) != Vector([1, 3])

    def test_equality_with_non_vector_returns_notimplemented(self):
        assert (Vector([1, 2]) == [1, 2]) is False

    def test_repr(self):
        assert repr(Vector([1, 2])) == "Vector([1, 2])"

    def test_copy_is_independent(self):
        v = Vector([1, 2, 3])
        c = v.copy()
        c[0] = 999
        assert v[0] == 1

    def test_unhashable(self):
        with pytest.raises(TypeError):
            hash(Vector([1, 2]))


# ----------------------------------------------------------------------
# Arithmetic
# ----------------------------------------------------------------------
class TestArithmetic:
    def test_add_vectors(self):
        assert (Vector([1, 2]) + Vector([3, 4])).components == [4, 6]

    def test_add_scalar(self):
        assert (Vector([1, 2]) + 10).components == [11, 12]

    def test_radd_scalar(self):
        assert (10 + Vector([1, 2])).components == [11, 12]

    def test_add_dimension_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            Vector([1, 2]) + Vector([1, 2, 3])

    def test_sub_vectors(self):
        assert (Vector([5, 5]) - Vector([1, 2])).components == [4, 3]

    def test_rsub_scalar(self):
        assert (10 - Vector([1, 2])).components == [9, 8]

    def test_sub_dimension_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            Vector([1, 2]) - Vector([1])

    def test_mul_scalar(self):
        assert (Vector([1, 2]) * 3).components == [3, 6]
        assert (3 * Vector([1, 2])).components == [3, 6]

    def test_mul_by_vector_raises_type_error(self):
        with pytest.raises(TypeError, match="dot"):
            Vector([1, 2]) * Vector([1, 2])

    def test_neg(self):
        assert (-Vector([1, -2])).components == [-1, 2]

    def test_truediv_scalar(self):
        assert (Vector([2, 4]) / 2).components == [1.0, 2.0]

    def test_truediv_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            Vector([1, 2]) / 0

    def test_truediv_by_nan_raises_value_error(self):
        with pytest.raises(ValueError):
            Vector([1, 2]).__truediv__(float("nan"))

    def test_truediv_non_numeric_raises_type_error(self):
        with pytest.raises(TypeError):
            Vector([1, 2]) / "x"

    def test_add_unsupported_type_returns_notimplemented(self):
        with pytest.raises(TypeError):
            Vector([1, 2]) + "x"


# ----------------------------------------------------------------------
# Dot / cross product -- correctness against NumPy
# ----------------------------------------------------------------------
class TestDotAndCross:
    def test_dot_matches_numpy(self):
        a, b = [1, 2, 3], [4, 5, 6]
        assert Vector(a).dot(Vector(b)) == pytest.approx(np.dot(a, b))

    def test_dot_empty_vectors_is_zero(self):
        assert Vector([]).dot(Vector([])) == 0

    def test_dot_dimension_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            Vector([1, 2]).dot(Vector([1, 2, 3]))

    def test_dot_non_vector_raises_type_error(self):
        with pytest.raises(TypeError):
            Vector([1, 2]).dot([1, 2])

    def test_cross_matches_numpy(self):
        a, b = [1, 0, 0], [0, 1, 0]
        result = Vector(a).cross(Vector(b)).components
        np.testing.assert_allclose(result, np.cross(a, b))

    def test_cross_random_vectors_match_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.uniform(-10, 10, 3)
        b = rng.uniform(-10, 10, 3)
        result = Vector(list(a)).cross(Vector(list(b))).components
        np.testing.assert_allclose(result, np.cross(a, b), atol=1e-9)

    def test_cross_wrong_dimension_raises(self):
        with pytest.raises(ValueError, match="3D"):
            Vector([1, 2]).cross(Vector([1, 2]))

    def test_cross_non_vector_raises_type_error(self):
        with pytest.raises(TypeError):
            Vector([1, 2, 3]).cross([1, 2, 3])


# ----------------------------------------------------------------------
# Norms
# ----------------------------------------------------------------------
class TestNorm:
    def test_l2_norm_matches_numpy(self):
        v = [3, 4]
        assert Vector(v).norm() == pytest.approx(np.linalg.norm(v))

    def test_l1_norm_matches_numpy(self):
        v = [3, -4, 5]
        assert Vector(v).norm(order=1) == pytest.approx(np.linalg.norm(v, ord=1))

    def test_linf_norm_matches_numpy(self):
        v = [3, -4, 5]
        assert Vector(v).norm(order=math.inf) == pytest.approx(
            np.linalg.norm(v, ord=np.inf)
        )

    def test_general_p_norm_matches_numpy(self):
        v = [1, 2, 3, 4]
        assert Vector(v).norm(order=3) == pytest.approx(np.linalg.norm(v, ord=3))

    def test_empty_vector_norm_is_zero(self):
        assert Vector([]).norm() == 0.0

    def test_invalid_norm_order_raises(self):
        with pytest.raises(ValueError):
            Vector([1, 2]).norm(order=-1)

    def test_invalid_norm_order_type_raises(self):
        with pytest.raises(ValueError):
            Vector([1, 2]).norm(order="bad")

    def test_abs_equals_l2_norm(self):
        v = Vector([3, 4])
        assert abs(v) == v.norm()


# ----------------------------------------------------------------------
# Normalize / is_zero
# ----------------------------------------------------------------------
class TestNormalize:
    def test_normalize_unit_length(self):
        v = Vector([3, 4]).normalize()
        assert v.norm() == pytest.approx(1.0)

    def test_normalize_zero_vector_raises(self):
        with pytest.raises(ZeroVectorError):
            Vector([0, 0]).normalize()

    def test_normalize_empty_vector_raises(self):
        with pytest.raises(ZeroVectorError):
            Vector([]).normalize()

    def test_normalize_with_tolerance(self):
        with pytest.raises(ZeroVectorError):
            Vector([1e-15, 1e-15]).normalize(tol=1e-9)

    def test_is_zero(self):
        assert Vector([0, 0, 0]).is_zero() is True
        assert Vector([0, 0.1]).is_zero() is False
        assert Vector([1e-10]).is_zero(tol=1e-9) is True


# ----------------------------------------------------------------------
# Angle / projection
# ----------------------------------------------------------------------
class TestAngleAndProjection:
    def test_angle_orthogonal_vectors(self):
        assert Vector([1, 0]).angle(Vector([0, 1])) == pytest.approx(90.0)

    def test_angle_parallel_vectors(self):
        assert Vector([1, 0]).angle(Vector([2, 0])) == pytest.approx(0.0)

    def test_angle_opposite_vectors(self):
        assert Vector([1, 0]).angle(Vector([-1, 0])) == pytest.approx(180.0)

    def test_angle_radians(self):
        result = Vector([1, 0]).angle(Vector([0, 1]), degrees=False)
        assert result == pytest.approx(math.pi / 2)

    def test_angle_with_zero_vector_raises(self):
        with pytest.raises(ZeroVectorError):
            Vector([1, 0]).angle(Vector([0, 0]))

    def test_angle_non_vector_raises_type_error(self):
        with pytest.raises(TypeError):
            Vector([1, 0]).angle([0, 1])

    def test_angle_floating_point_drift_clamped(self):
        # Nearly-parallel vectors that could push cos slightly outside
        # [-1, 1] due to floating point error must not raise a domain error.
        v = Vector([1.0, 1e-16])
        result = v.angle(v)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_projection_onto_matches_expected(self):
        # Projection of (3, 4) onto (1, 0) is (3, 0).
        result = Vector([3, 4]).projection_onto(Vector([1, 0]))
        assert result.components == pytest.approx([3.0, 0.0])

    def test_projection_onto_zero_vector_raises(self):
        with pytest.raises(ZeroVectorError):
            Vector([1, 2]).projection_onto(Vector([0, 0]))

    def test_projection_non_vector_raises_type_error(self):
        with pytest.raises(TypeError):
            Vector([1, 2]).projection_onto([0, 0])


# ----------------------------------------------------------------------
# Distance
# ----------------------------------------------------------------------
class TestDistance:
    def test_distance_matches_numpy(self):
        a, b = [1, 2, 3], [4, 0, -1]
        expected = np.linalg.norm(np.array(a) - np.array(b))
        assert Vector(a).distance_to(Vector(b)) == pytest.approx(expected)

    def test_distance_dimension_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            Vector([1, 2]).distance_to(Vector([1, 2, 3]))

    def test_distance_to_self_is_zero(self):
        v = Vector([1, 2, 3])
        assert v.distance_to(v) == 0.0


# ----------------------------------------------------------------------
# Element-wise helpers
# ----------------------------------------------------------------------
class TestElementWise:
    def test_element_wise(self):
        result = Vector([1, 2, 3]).element_wise(lambda x: x ** 2)
        assert result.components == [1, 4, 9]

    def test_element_wise_with(self):
        result = Vector([1, 2, 3]).element_wise_with(Vector([4, 5, 6]), lambda a, b: a + b)
        assert result.components == [5, 7, 9]

    def test_element_wise_with_dimension_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            Vector([1, 2]).element_wise_with(Vector([1, 2, 3]), lambda a, b: a + b)

    def test_element_wise_with_non_vector_raises_type_error(self):
        with pytest.raises(TypeError):
            Vector([1, 2]).element_wise_with([1, 2], lambda a, b: a + b)


# ----------------------------------------------------------------------
# Interop / constructors
# ----------------------------------------------------------------------
class TestInterop:
    def test_to_list_returns_plain_list(self):
        v = Vector([1, 2, 3])
        lst = v.to_list()
        assert lst == [1, 2, 3]
        assert type(lst) is list

    def test_to_numpy_matches_components(self):
        v = Vector([1, 2, 3])
        np.testing.assert_allclose(v.to_numpy(), np.array([1.0, 2.0, 3.0]))

    def test_zeros_constructor(self):
        v = Vector.zeros(4)
        assert v.components == [0.0, 0.0, 0.0, 0.0]

    def test_zeros_negative_raises(self):
        with pytest.raises(ValueError):
            Vector.zeros(-1)

    def test_from_list(self):
        v = Vector.from_list([1, 2, 3])
        assert v.components == [1, 2, 3]


# ----------------------------------------------------------------------
# NaN / Inf propagation (documented, non-raising behavior consistent w/ NumPy)
# ----------------------------------------------------------------------
class TestNanInfPropagation:
    def test_dot_with_nan_propagates_nan(self):
        result = Vector([1, float("nan")]).dot(Vector([1, 1]))
        assert math.isnan(result)

    def test_norm_with_inf_component_is_inf(self):
        assert Vector([1, float("inf")]).norm() == math.inf

    def test_norm_matches_numpy_with_inf(self):
        v = [1.0, float("inf")]
        assert Vector(v).norm() == pytest.approx(np.linalg.norm(v))
