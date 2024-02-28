from typing import List, Optional, Tuple

import cvxopt
import numpy as np
import scipy
import scipy.spatial
from cvxopt import solvers
from scipy.optimize import linprog


class Polytope:
    def __init__(
        self,
        A_norm: np.ndarray,
        b_norm: np.ndarray,
        feature_mean: Optional[np.ndarray] = None,
        feature_scale: Optional[np.ndarray] = None,
    ):
        self.A_norm = A_norm
        self.b_norm = b_norm

        if feature_mean is not None:
            assert feature_scale is not None
            self.A = A_norm / (feature_scale).reshape((1, -1))
            self.b = b_norm + A_norm @ (feature_mean / feature_scale)
            self.feature_mean: Optional[np.ndarray] = feature_mean
            self.feature_scale: Optional[np.ndarray] = feature_scale
        else:
            self.A = A_norm
            self.b = b_norm
            self.feature_mean: Optional[np.ndarray] = None
            self.feature_scale: Optional[np.ndarray] = None

    def normalize(self, x: np.ndarray):
        if self.feature_mean is None or self.feature_scale is None:
            return x
        return (x - self.feature_mean) / self.feature_scale


def _get_adjacent_facets(simplices: np.ndarray, point: int) -> np.ndarray:
    """Returns the indices of the simplices that contain the given point."""
    return np.where(np.any(simplices == point, axis=1))[0]


def get_min_max_features(polytope: Polytope) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the possible range of each feature in a given polytope.

    Runs an LP to determine the minimum and maximum each feature can be in the polytope.

    Args:
        polytope: The polytope for which to compute the min and max feature values.

    Returns:
        min_features: A 1D array of the minimum values for each feature.
        max_features: A 1D array of the maximum values for each feature.

    Raises:
        ValueError: If a linear program cannot be solved.
    """
    A, b = polytope.A, polytope.b
    num_features = A.shape[1]
    min_features = np.zeros(num_features)
    max_features = np.zeros(num_features)
    for f in range(num_features):
        c = np.zeros(num_features)
        c[f] = 1
        res = linprog(c, A_ub=A, b_ub=b)
        min_features[f] = res.fun
        c[f] = -1
        res = linprog(c, A_ub=A, b_ub=b)
        max_features[f] = -res.fun
    return min_features, max_features


def find_closest_point_in_polytope(point: np.ndarray, polytope: Polytope) -> float:
    """Finds the closest point in a given polytope to the provided point using QP.

    Solves
        minimize (x - point)**2
        such that x in polytope

    Args:
        point: The point for which we want to find the closest point in the polytope.
        polytope: The polytope in which to search for the closest point.

    Returns:
        The minimum squared distance from the input point to any point in the polytope.
    """
    # Compute distance in normalized space to make it independent of feature scales
    point = polytope.normalize(point)

    # Objective: minimize (x - point)**2 = x.T x - 2 point.T x
    (n,) = point.shape
    P = cvxopt.matrix(2 * np.identity(n), tc="d")
    q = cvxopt.matrix(-2 * point, tc="d")

    # Constraints: G * x <= h
    G = cvxopt.matrix(polytope.A_norm, tc="d")
    h = cvxopt.matrix(polytope.b_norm, tc="d")

    initvals = {"x": cvxopt.matrix(point.reshape((n, 1)))}

    # Solve the QP
    solvers.options["show_progress"] = False
    cvxopt.solvers.options["maxiters"] = 1000
    cvxopt.solvers.options["abstol"] = 1e-7
    cvxopt.solvers.options["reltol"] = 1e-6
    solution = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, initvals=initvals)

    x = np.array(solution["x"]).reshape((n,))
    return np.sum(np.square(x - point))


def _furthest_point(points: np.ndarray, polytope: Polytope) -> Tuple[int, float]:
    """Finds the point in a given set that is furthest from a given polytope.

    Iterates over a collection of points and for each point, computes the closest point
    in the polytope using 'find_closest_point_in_polytope' function.

    Args:
        points: A 2D array where each row represents a point in the space.
        polytope: The polytope from which distances are to be calculated.

    Returns:
        max_i: The index of the point that is furthest from the polytope.
        max_dist: The squared distance of the returned point from the polytope.
    """
    max_dist = -float("inf")
    max_i = None

    for i, point in enumerate(points):
        dist = find_closest_point_in_polytope(point, polytope)
        if dist > max_dist:
            max_dist, max_i = dist, i

    return max_i, max_dist


def is_in_polytope(
    point: np.ndarray, polytope: Polytope, inclusive: bool = True, eps: float = 1e-6
) -> bool:
    """Determines whether a given point is inside a specified polytope.

    If `inclusive` is True compute the distance to the polytope using QP and compare it
    to eps. This is more robust but more expensive.

    Otherwise, just check the inequality `A @ point <= b` exactly.

    Args:
        point: The point to be checked.
        polytope: The polytope in which to check for the presence of the point.
        inclusive: Whether to use QP to determine distance to polytope instead of
            checking the inequlity exactly.
        eps: Threshold for distance to polytope being considered inside. Ignored if
            inclusive is False.

    Returns:
        A boolean indicating whether the point is in the polytope.
    """
    if inclusive:
        min_val = find_closest_point_in_polytope(point, polytope)
        return min_val <= eps

    return bool(np.all(polytope.A @ point <= polytope.b))


def get_safe_set(
    corners: np.ndarray,
    normalize: bool = True,
    num_points: Optional[int] = None,
    stopping_dist: float = 1e-6,
    min_singular_value: float = 1e-6,
    orthogonal_tolerance: float = 1e-10,
    duplicate_precision: int = 8,
    starting_set: List[int] = [],
    return_vertices: bool = False,
) -> Tuple[Polytope, List[Polytope]]:
    """Iteratively construct safe set as convex hull of corners.

    Uses a greedy method to construct convex hull: start with 3 points and iteratively
    add the point that has the largest distance to the other points until the distance
    is lower than `stopping_dist`.

    Args:
        corners: Points to use for constructing safe set.
        normalize: Whether to normalize the points before constructing safe set.
        num_points: Maximum number of points to choose for the convex hull.
        stopping_dist

    Returns:
        Safe set and list of unsafe sets.
    """

    corners = np.copy(corners)

    _, unique_idx = np.unique(
        np.round(corners, duplicate_precision), axis=0, return_index=True
    )
    corners = corners[unique_idx]
    num_corners, num_dims = corners.shape

    if num_corners == 1:
        print("Special case: 1 corner")
        A = np.vstack([np.eye(num_dims), -np.eye(num_dims)])
        b = np.concatenate([corners[0], -corners[0]])
        if return_vertices:
            return Polytope(A, b), [], []
        return Polytope(A, b), []
    if num_corners == 2:
        print("Special case: 2 corners")
        vec = (corners[1] - corners[0]).reshape((1, -1))
        # find d-1 orthogonal vectors
        _, _, V = np.linalg.svd(vec)
        removed_components = V[1:]
        # orthogonal components must be zero
        # vec components must be between the two points
        removed_vec = removed_components @ corners[0]
        A = np.vstack([removed_components, -removed_components, vec, -vec])
        b = np.concatenate(
            [
                removed_vec
                + orthogonal_tolerance * np.ones(removed_components.shape[0]),
                -removed_vec
                + orthogonal_tolerance * np.ones(removed_components.shape[0]),
                vec @ corners[1],
                -vec @ corners[0],
            ]
        )
        if return_vertices:
            return Polytope(A, b), [], []
        return Polytope(A, b), []

    print(f"Num points: {num_points}   ({num_corners} corners)")

    if num_points is None or num_corners <= num_points:
        num_points = num_corners

    point_count = max(3 - len(starting_set), 0)
    remaining_idx = np.arange(num_corners)
    point_idx = np.random.choice(remaining_idx, point_count, replace=False)
    if len(starting_set) > 0:
        point_idx = np.concatenate([point_idx, starting_set])
    remaining_idx = np.array([i for i in remaining_idx if i not in point_idx])

    while point_count < num_points:
        polytope, _ = _get_safe_set_from_fixed_points(
            corners=corners[point_idx],
            normalize=normalize,
            min_singular_value=min_singular_value,
            orthogonal_tolerance=orthogonal_tolerance,
            return_vertices=False,
        )
        add_idx, add_dist = _furthest_point(corners[remaining_idx], polytope)
        add_idx = remaining_idx[add_idx]
        if add_dist < stopping_dist:
            print(f"Distance of furthest point {add_dist} < {stopping_dist}.")
            print(f"Stopping with {point_count} points.")
            break
        point_idx = np.concatenate([point_idx, [add_idx]])
        remaining_idx = np.array([i for i in remaining_idx if i not in point_idx])
        point_count += 1
        print(f"Added point {add_idx}  (dist: {add_dist})")
        print("point_idx:", point_idx)

    return _get_safe_set_from_fixed_points(
        corners=corners[point_idx],
        normalize=normalize,
        min_singular_value=min_singular_value,
        orthogonal_tolerance=orthogonal_tolerance,
        return_vertices=return_vertices,
    )


def _get_safe_set_from_fixed_points(
    corners: np.ndarray,
    normalize: bool,
    min_singular_value: float,
    orthogonal_tolerance: float,
    return_vertices: bool,
) -> Tuple[Polytope, List[Polytope]]:
    """Construct safe set from a fixed set of corners.

    Computes the convex hull of the corners using qhull (http://www.qhull.org/)

    Normalizes the date if `normalize` is True.

    If data is not full rank, it is projected onto a lower dimensional subspace
    and the resulting convex hull is projected back to the full space.

    Attributes:
        corners: Points to use for constructing safe set.
        normalize: Whether to normalize the points before constructing safe set.
        min_singular_value: Used to determine effective dimensionality of corners-
        orthogonal_tolerance: If convex hull is determined in a lower dimensional
            subspace, this tolerance controls the size of the safe set in the
            orthogonal directions.

    Returns:
        Safe set and list of unsafe sets.
    """

    if normalize:
        # normalize corners
        feature_mean = np.mean(corners, axis=0)
        feature_scale = np.std(corners, axis=0)

        # avoid division by zero
        feature_scale[feature_scale <= 0.01] = 1
        corners = (corners - feature_mean) / feature_scale
    else:
        feature_mean, feature_scale = None, None

    U, d, V = np.linalg.svd((corners - corners[-1])[:-1], full_matrices=True)

    num_dim = max(np.sum(d > min_singular_value), 2)  # qhull needs at least 2-D
    print("Effective dimension", num_dim)

    if num_dim < corners.shape[1] + 1:
        # 1. project corners to lower dimensional space
        projection = V[:num_dim]
        removed_components = V[num_dim:]
        projected_corners = corners @ projection.T

        # 2. construct convex hull in that space
        ch = scipy.spatial.ConvexHull(projected_corners, qhull_options="QJ")
        A = ch.equations[:, :-1]
        b = -ch.equations[:, -1]

        # 3. project A, b back to full space
        A = A @ projection

        # 4. Add constraints to make the convex hull bounded in the full space
        removed_vec = removed_components @ corners[0]
        A_additional = np.vstack((removed_components, -removed_components))
        b_additional = np.concatenate(
            [removed_vec + orthogonal_tolerance, -(removed_vec - orthogonal_tolerance)]
        )

        A = np.vstack((A, A_additional))
        b = np.hstack((b, b_additional))

        safe_polytope = Polytope(
            A_norm=A,
            b_norm=b,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )

        unsafe_polytopes = []

        num_points = len(ch.points)
        for point in range(num_points):
            adj_facets = _get_adjacent_facets(ch.simplices, point)
            if len(adj_facets) == 0:
                continue

            A = -ch.equations[adj_facets, :-1]
            b = ch.equations[adj_facets, -1]

            A = A @ projection
            A = np.vstack((A, A_additional))
            b = np.hstack((b, b_additional))

            unsafe_polytope = Polytope(
                A_norm=A,
                b_norm=b,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
            )
            unsafe_polytopes.append(unsafe_polytope)

        if return_vertices:
            points = ch.points @ projection
            if normalize:
                points = points * feature_scale + feature_mean
            return safe_polytope, unsafe_polytopes, points

        return safe_polytope, unsafe_polytopes

    else:
        ch = scipy.spatial.ConvexHull(corners, qhull_options="QJ")
        A = ch.equations[:, :-1]
        b = -ch.equations[:, -1]

        safe_polytope = Polytope(
            A_norm=A,
            b_norm=b,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )

        unsafe_polytopes = []

        num_points = len(ch.points)
        for point in range(num_points):
            adj_facets = _get_adjacent_facets(ch.simplices, point)
            if len(adj_facets) == 0:
                continue
            A = -ch.equations[adj_facets, :-1]
            b = ch.equations[adj_facets, -1]
            unsafe_polytope = Polytope(
                A_norm=A,
                b_norm=b,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
            )
            unsafe_polytopes.append(unsafe_polytope)

        if return_vertices:
            points = ch.points
            if feature_scale is not None:
                points = points * feature_scale + feature_mean
            return safe_polytope, unsafe_polytopes, points

        return safe_polytope, unsafe_polytopes
