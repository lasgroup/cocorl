import numpy as np
import pytest

from constraint_learning.linear import algorithm


@pytest.mark.parametrize(
    "num_dims, num_corners",
    [
        (2, 3),
        (2, 5),
        (4, 5),
        (6, 3),
        (6, 20),
        (9, 5),
        (9, 20),
    ],
)
@pytest.mark.parametrize("num_duplicates", [0, 1])
@pytest.mark.parametrize("scale", [1, 100])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_corners_in_safe_set(num_dims, num_corners, num_duplicates, scale, seed):
    # sample random corners
    np.random.seed(seed)
    corners = np.random.uniform(low=-scale, high=scale, size=(num_corners, num_dims))
    corners = np.concatenate([corners] + [corners] * num_duplicates)

    # Construct safe set
    safe_polytope, _ = algorithm.get_safe_set(corners, normalize=True)

    tolerance = 1e-5

    # All points used to construct the safe set should be inside the safe set
    for corner in corners:
        d = algorithm.find_closest_point_in_polytope(corner, safe_polytope)
        assert d <= tolerance
        assert algorithm.is_in_polytope(
            corner, safe_polytope, inclusive=True, eps=tolerance
        )

    # Convex combination of the points should also be in safe set
    convex_combination = np.mean(corners, axis=0)
    d = algorithm.find_closest_point_in_polytope(convex_combination, safe_polytope)
    assert d <= tolerance
    assert algorithm.is_in_polytope(
        convex_combination, safe_polytope, inclusive=True, eps=tolerance
    )


@pytest.mark.parametrize("scale", [1, 10, 100])
@pytest.mark.parametrize("num_dims", [4, 6, 8, 10])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_min_max_features(scale, num_dims, seed):
    tolerance = 1e-4
    num_samples = 20

    np.random.seed(seed)
    corners = np.random.uniform(low=-scale, high=scale, size=(num_samples, num_dims))
    polytope, _ = algorithm.get_safe_set(corners)
    min_features, max_features = algorithm.get_min_max_features(polytope)

    np.testing.assert_array_less(corners.min(axis=0), min_features + tolerance)
    np.testing.assert_array_less(max_features, corners.max(axis=0) + tolerance)
