from typing import Optional

import numpy as np
import sacred
import scipy

from constraint_learning import util
from constraint_learning.linear import algorithm
from constraint_learning.util import logging, results

ex = sacred.Experiment("synthetic_experiment")
ex.observers = [logging.SetID(), sacred.observers.FileStorageObserver("results")]


def solve_lp(*, theta: np.ndarray, A: np.ndarray, b: np.ndarray, bounds: list):
    """Minimizes theta_i^T x under constraints A, b for each theta_i in theta."""
    res = scipy.optimize.linprog(
        c=-theta,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
    )

    if res.success:
        return res.x, True
    return np.zeros_like(theta), False


@ex.config
def cfg():
    num_dims = 2
    num_thetas = 5
    num_new_thetas = 10
    num_phis = 8
    max_thetas = 100
    new_theta_seed = None
    seed = 1
    active_learning = False
    experiment_label = None


@ex.automain
def synthetic_experiment(
    num_dims: int,
    num_thetas: int,
    num_new_thetas: int,
    num_phis: int,
    max_thetas: int,
    new_theta_seed: Optional[int],
    active_learning: bool,
    seed: int,
    experiment_label: str,
) -> results.SyntheticExperimentResult:
    xmin, xmax = -5, 5

    if num_thetas > max_thetas:
        print(
            "Warning, max_thetas too low. "
            f"num_thetas: {num_thetas}; max_thetas: {max_thetas}"
        )
        max_thetas = num_thetas

    # sample random constraints
    new_theta = util.sampling.sample_unit_vectors(
        num_samples=num_new_thetas, num_dims=num_dims
    )
    phi = util.sampling.sample_unit_vectors(num_samples=num_phis, num_dims=num_dims)
    threshold = np.ones(num_phis)

    if active_learning:
        theta = []
        demonstrations = []
        unsafe_polytopes = []
        vertices = []

        for i in range(num_thetas):
            print("Active learning iteration", i)

            num_candidate_thetas = 100
            active_learning_strength = 20

            logits = np.zeros(num_candidate_thetas)

            cand_theta = util.sampling.sample_unit_vectors(
                num_samples=num_candidate_thetas, num_dims=num_dims
            )

            if i == 0:
                th = cand_theta[0]
            else:
                A = safe_polytope.A  # noqa: F821
                A /= np.linalg.norm(A, axis=1, keepdims=True)

                dots = cand_theta @ A.T
                angles = np.arccos(dots)
                logits = -np.min(angles, axis=1)

                print("logits", logits)
                logits = active_learning_strength * logits
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()

                print("probs", probs)

                th = cand_theta[np.random.choice(num_candidate_thetas, p=probs)]

            theta.append(th)

            res, found_solution = solve_lp(
                theta=th, A=phi, b=threshold, bounds=[(xmin, xmax)] * num_dims
            )
            assert found_solution
            demonstrations.append(res)

            safe_polytope, unsafe_polytopes, vertices = algorithm.get_safe_set(
                demonstrations,
                normalize=True,
                num_points=None,
                stopping_dist=1e-12,
                orthogonal_tolerance=0,
                duplicate_precision=12,
                return_vertices=True,
            )
    else:
        theta = util.sampling.sample_unit_vectors(
            num_samples=max_thetas, num_dims=num_dims
        )
        theta = theta[:num_thetas]

        # get demonstrations
        demonstrations = []
        for th in theta:
            res, found_solution = solve_lp(
                theta=th, A=phi, b=threshold, bounds=[(xmin, xmax)] * num_dims
            )
            assert found_solution
            demonstrations.append(res)

        # get safe set
        safe_polytope, unsafe_polytopes = algorithm.get_safe_set(
            demonstrations,
            normalize=True,
            num_points=None,
            stopping_dist=1e-12,
            orthogonal_tolerance=0,
            duplicate_precision=12,
        )

    # compute size of safe set over bounded domain
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(xmin, xmax, 100)

    num_samples = 100000
    x = np.random.uniform(low=xmin, high=xmax, size=(num_samples, num_dims))

    A, b = safe_polytope.A, safe_polytope.b
    safe_set = np.all(x @ A.T <= b, axis=1)
    safe_set_size = np.mean(safe_set)

    unsafe_set = np.zeros((num_samples,), dtype=bool)
    for polytope in unsafe_polytopes:
        A, b = polytope.A, polytope.b
        unsafe_set_poly = np.all(x @ A.T <= b, axis=1)
        unsafe_set |= unsafe_set_poly
    unsafe_set_size = np.mean(unsafe_set)

    uncertain_set_size = 1 - safe_set_size - unsafe_set_size

    safe_reward = np.zeros(len(new_theta))
    true_reward = np.zeros(len(new_theta))
    min_reward = np.zeros(len(new_theta))
    found_safe_solution = np.zeros(len(new_theta))
    safe_constraint_violations = np.zeros(len(new_theta))
    safe_constraint_distance = np.zeros(len(new_theta))
    safe_inferred_constraint_violations = np.zeros(len(new_theta))
    safe_inferred_constraint_distance = np.zeros(len(new_theta))
    true_solution_in_safe_set = np.zeros(len(new_theta), dtype=bool)
    true_solution_in_unsafe_set = np.zeros(len(new_theta), dtype=bool)
    true_solution_in_uncertain_set = np.zeros(len(new_theta), dtype=bool)

    for i, new_theta_i in enumerate(new_theta):
        safe_solution, found_solution = solve_lp(
            theta=new_theta_i,
            A=safe_polytope.A,
            b=safe_polytope.b,
            bounds=[(xmin, xmax)] * num_dims,
        )
        found_safe_solution[i] = found_solution

        true_solution, found_true = solve_lp(
            theta=new_theta_i, A=phi, b=threshold, bounds=[(xmin, xmax)] * num_dims
        )
        assert found_true

        worst_solution, found_worst = solve_lp(
            theta=-new_theta_i, A=phi, b=threshold, bounds=[(xmin, xmax)] * num_dims
        )
        assert found_worst

        safe_reward[i] = np.dot(safe_solution, new_theta_i)
        true_reward[i] = np.dot(true_solution, new_theta_i)
        min_reward[i] = np.dot(worst_solution, new_theta_i)

        safe_constraint_violations[i] = np.sum(
            np.maximum(phi @ safe_solution - threshold, 0)
        )

        safe_constraint_distance[i] = algorithm.find_closest_point_in_polytope(
            safe_solution, algorithm.Polytope(phi, threshold)
        )

        safe_inferred_constraint_violations[i] = np.sum(
            np.maximum(A @ safe_solution - b, 0)
        )

        safe_inferred_constraint_distance[i] = algorithm.find_closest_point_in_polytope(
            safe_solution, safe_polytope
        )

        true_solution_in_safe_set[i] = algorithm.is_in_polytope(
            true_solution, safe_polytope, inclusive=True
        )

        true_solution_in_unsafe_set[i] = False
        for unsafe_polytope in unsafe_polytopes:
            if algorithm.is_in_polytope(
                true_solution, unsafe_polytope, inclusive=False
            ):
                true_solution_in_unsafe_set[i] = True
                break

        true_solution_in_uncertain_set[i] = not (
            true_solution_in_safe_set[i] or true_solution_in_unsafe_set[i]
        )

    return results.SyntheticExperimentResult(
        true_reward=true_reward,
        safe_reward=safe_reward,
        min_reward=min_reward,
        found_safe_solution=found_safe_solution,
        safe_constraint_violations=safe_constraint_violations,
        safe_constraint_distance=safe_constraint_distance,
        safe_inferred_constraint_violations=safe_inferred_constraint_violations,
        safe_inferred_constraint_distance=safe_inferred_constraint_distance,
        true_solution_in_safe_set=true_solution_in_safe_set,
        true_solution_in_unsafe_set=true_solution_in_unsafe_set,
        true_solution_in_uncertain_set=true_solution_in_uncertain_set,
        safe_set_size=float(safe_set_size),
        unsafe_set_size=float(unsafe_set_size),
        uncertain_set_size=float(uncertain_set_size),
    )
