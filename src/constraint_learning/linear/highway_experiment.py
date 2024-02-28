"""Constraing learning experiment using cross entropy solver on highway_env."""
import glob
import itertools
import json
import random
from typing import Any, Dict, List, Optional

import numpy as np
import sacred

from constraint_learning.algos import cross_entropy
from constraint_learning.linear import algorithm
from constraint_learning.linear.irl import max_ent_cem
from constraint_learning.util import logging, results, sampling

TOLERANCE = 1e-5


# Indices of the features that are rewards
REWARD_INDICES = np.array([0, 1, 2, 3, 4])

# Indices of the features that are constraints.
CONSTRAINT_INDICES = np.array([5, 6, 7, 8])


Demonstration = Dict[str, Any]

ex = sacred.Experiment("highway_experiment_ce")
ex.observers = [logging.SetID(), sacred.observers.FileStorageObserver("results")]


def solve_with_reinits(
    *,
    solver: cross_entropy.CrossEntropySolver,
    ce_solver_kwargs: dict,
    reward_parameters: np.ndarray,
    constraint_parameters: Optional[np.ndarray],
    constraint_thresholds: Optional[np.ndarray],
    reinits: int,
):
    results = [
        solver.solve(
            reward_parameters=reward_parameters,
            constraint_parameters=constraint_parameters,
            constraint_thresholds=constraint_thresholds,
            callback=None,
            **ce_solver_kwargs,
        )
        for _ in range(reinits)
    ]

    feasible = any([r.feasible for r in results])

    features = None
    best_rew = -np.inf
    if feasible:
        for result in results:
            if result.feasible and (
                features is None or result.features @ reward_parameters > best_rew
            ):
                features = result.features
                best_rew = result.features @ reward_parameters
    else:
        features = results[0].features

    return feasible, features


def sample_demos(
    all_demos: List[Demonstration],
    num_samples: int,
    env_name: str,
    allowed_goals: List[str],
    seed: Optional[int] = None,
):
    """Samples demonstrations from `env_name` with goals in `allowed_goals`."""
    goal_idx = {
        {
            "left": 0,
            "middle": 1,
            "right": 2,
        }[goal]
        for goal in allowed_goals
    }
    all_demos_shuffled = all_demos.copy()

    with sampling.temp_seed(seed):
        random.shuffle(all_demos_shuffled)
    shuffled_demos_iter = itertools.cycle(iter(all_demos_shuffled))

    sample: List[Demonstration] = []
    while len(sample) < num_samples:
        demo = next(shuffled_demos_iter)
        if demo["env_name"] == env_name and np.argmax(demo["goal"]) in goal_idx:
            sample.append(demo)
    return sample


@ex.named_config
def debug():
    ce_solver_kwargs = {
        "iterations": 1,
        "num_candidates": 2,
        "num_elite": 1,
        "num_trajectories": 2,
        "verbose": True,
    }


@ex.named_config
def fast():
    ce_solver_kwargs = {
        "iterations": 20,
        "num_candidates": 100,
        "num_elite": 20,
        "num_trajectories": 10,
        "verbose": True,
    }


@ex.config
def cfg():
    num_thetas = 5
    num_new_thetas = 5
    source_goals = ["left", "middle", "right"]
    target_goals = ["left", "middle", "right"]
    source_env = "Intersect-TruncateOnly-v0"
    target_env = "Intersect-TruncateOnly-v0"
    demonstration_folder = "demonstrations/"
    restrict_to_constraint_features = False
    seed = 1
    experiment_label = None
    ce_solver_kwargs = {
        "iterations": 30,
        "num_candidates": 500,
        "num_elite": 20,
        "num_trajectories": 30,
        "verbose": True,
    }
    env_config = {
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "duration": 15,
    }
    num_jobs = 20
    solver_reinit = 1
    add_non_moving_demo = False
    method = "constraint_learning"
    new_theta_seed = None
    irl_min_iterations = 1


@ex.automain
def highway_experiment(
    _run,
    num_thetas: int,
    num_new_thetas: int,
    source_goals: List[str],
    target_goals: List[str],
    source_env: str,
    target_env: str,
    demonstration_folder: str,
    restrict_to_constraint_features: bool,
    seed: int,
    experiment_label: str,
    ce_solver_kwargs: dict,
    env_config: dict,
    num_jobs: int,
    solver_reinit: int,
    add_non_moving_demo: bool,
    method: str,
    new_theta_seed: Optional[int],
    irl_min_iterations: int,
) -> results.CEHighwayExperimentResult:
    print(f"Experiment {experiment_label} (Seed: {seed})")

    demonstration_files = glob.glob(f"{demonstration_folder}/*.json")
    phi, threshold = None, None

    all_demos = []
    for filename in demonstration_files:
        with open(filename, "r") as f:
            demo = json.load(f)

            if phi is None:
                phi = np.array(demo["constraint_parameters"])
                threshold = np.array(demo["constraint_thresholds"])
            assert np.allclose(phi, demo["constraint_parameters"]), (
                phi,
                demo["constraint_parameters"],
            )
            assert np.allclose(threshold, demo["constraint_thresholds"]), (
                threshold,
                demo["constraint_thresholds"],
            )

            features = np.array(demo["features"])
            if np.all(phi @ features <= threshold):
                all_demos.append(demo)
            else:
                print(f"Skipping demonstration {filename} because it is infeasible.")

    source_sample = sample_demos(
        all_demos=all_demos,
        num_samples=num_thetas,
        env_name=source_env,
        allowed_goals=source_goals,
    )
    theta = [d["reward_parameters"] for d in source_sample]
    demonstrations = [d["features"] for d in source_sample]

    # hacky way of adding a "safe" demonstrations that is not moving. Turns out
    # that in the intersection environment this corresponds to adding a zero vector.
    # For other environments, we'd have to create such a demonstration explicitly
    if add_non_moving_demo:
        theta.append(np.zeros(9))
        demonstrations.append(np.zeros(9))
        starting_set = [len(demonstrations) - 1]
    else:
        starting_set = []

    theta = np.array(theta)
    demonstrations = np.array(demonstrations)

    target_sample = sample_demos(
        all_demos=all_demos,
        num_samples=num_new_thetas,
        env_name=target_env,
        allowed_goals=target_goals,
        seed=new_theta_seed,
    )
    new_theta = np.array([d["reward_parameters"] for d in target_sample])
    true_solutions = np.array([d["features"] for d in target_sample])

    _, unique_idx = np.unique(np.round(demonstrations, 6), axis=0, return_index=True)

    print("Thetas:")
    for th in theta:
        print(th)
    print("Demonstrations:")
    for d in demonstrations:
        print(d)
    print("Unique demonstrations:", unique_idx)

    source_solver = cross_entropy.CrossEntropySolver(
        source_env, env_config, num_jobs=num_jobs
    )
    target_solver = cross_entropy.CrossEntropySolver(
        target_env, env_config, num_jobs=num_jobs
    )

    if method == "constraint_learning":
        # get safe set
        safe_set_kwargs = {
            "stopping_dist": 1e-8,
            "min_singular_value": 1e-8,
            "orthogonal_tolerance": 1e-8,
            "duplicate_precision": 8,
            "starting_set": starting_set,
        }

        if restrict_to_constraint_features:
            # get safe set restricted to constraint features
            safe_polytope, unsafe_polytopes = algorithm.get_safe_set(
                demonstrations[:, CONSTRAINT_INDICES], **safe_set_kwargs
            )

            # project safe set back to full space
            safe_polytope = algorithm.Polytope(
                safe_polytope.A @ np.identity(9)[CONSTRAINT_INDICES],
                safe_polytope.b,
            )
            unsafe_polytopes = [
                algorithm.Polytope(
                    polytope.A @ np.identity(9)[CONSTRAINT_INDICES], polytope.b
                )
                for polytope in unsafe_polytopes
            ]
        else:
            safe_polytope, unsafe_polytopes = algorithm.get_safe_set(
                demonstrations, **safe_set_kwargs
            )

        A, b = np.copy(safe_polytope.A), np.copy(safe_polytope.b)

        # solve for new rewards under inferred constraints
        found_safe_solution = []
        safe_solutions = []
        for th_i, th in enumerate(new_theta):
            feasible, features = solve_with_reinits(
                solver=target_solver,
                ce_solver_kwargs=ce_solver_kwargs,
                reward_parameters=th,
                constraint_parameters=A,
                constraint_thresholds=b,
                reinits=solver_reinit,
            )
            found_safe_solution.append(feasible)
            safe_solutions.append(features)

    else:
        # Run CEM with same parameters inside IRL but without reinits

        # For now IRL automatically splits reward and constraint features
        # which is similar to restricting the constraint learning algorithm
        # to only the constraint features.

        if method == "known_reward_irl_max_ent":
            irl_kwargs = {
                "learn_constraint": True,
                "learn_reward": False,
                "known_rewards": theta,
            }
        elif method == "shared_reward_irl_max_ent":
            irl_kwargs = {
                "learn_constraint": True,
                "learn_reward": True,
                "known_rewards": None,
            }
        elif method == "vanilla_irl_max_ent":
            irl_kwargs = {
                "learn_constraint": False,
                "learn_reward": True,
                "known_rewards": None,
            }
        else:
            raise NotImplementedError(f"Unknown method {method}")

        def callback(locals, globals):
            expert_features = locals["expert_features"]
            features = locals["features"]
            demo_i = locals["demo_i"]
            it = locals["it"]
            grad = locals["grad"]

            print("IRL iteration", it)
            print("expert_features", expert_features)
            print("features", features)
            print("grad", grad)

            expert_rew = np.dot(expert_features, theta[demo_i])
            current_rew = np.dot(features, theta[demo_i])
            regret = expert_rew - current_rew
            print("regret", regret)

            constraint = phi @ features - threshold
            print("constraint", constraint)

            _run.log_scalar("irl_regret", regret)
            _run.log_scalar("irl_constraint_total", constraint)

            for i, c_i in enumerate(constraint):
                _run.log_scalar(f"irl_constraint_{i}", c_i)

            for i in range(len(grad)):
                _run.log_scalar(f"irl_features_{i}", features[i])
                _run.log_scalar(f"irl_grad_{i}", grad[i])

        num_iterations = max(irl_min_iterations, num_thetas)

        inferred_rewards, inferred_constraint = max_ent_cem.max_ent_irl(
            solver=source_solver,
            demonstration_features=demonstrations,
            num_iterations=num_iterations,
            lr=1.0,
            num_rew_features=5,
            num_constraint_features=4,
            regularizer=0.0001,
            ce_solver_kwargs=ce_solver_kwargs,
            callback=callback,
            **irl_kwargs,
        )

        if method == "vanilla_irl_max_ent":
            irl_reward = inferred_rewards.mean(axis=0)
        else:
            irl_reward = inferred_constraint

        # solve for new rewards under inferred constraints
        found_safe_solution = []
        safe_solutions = []
        for th_i, th in enumerate(new_theta):
            print(f"Evaluate for theta {th_i}")
            feasible, features = solve_with_reinits(
                solver=target_solver,
                ce_solver_kwargs=ce_solver_kwargs,
                reward_parameters=th + irl_reward,
                constraint_parameters=None,
                constraint_thresholds=None,
                reinits=solver_reinit,
            )
            found_safe_solution.append(feasible)
            safe_solutions.append(features)

    safe_reward = np.zeros(len(new_theta))
    true_reward = np.zeros(len(new_theta))
    safe_constraint_violations = np.zeros(len(new_theta))
    true_solution_in_safe_set = np.zeros(len(new_theta), dtype=bool)
    true_solution_in_unsafe_set = np.zeros(len(new_theta), dtype=bool)
    true_solution_in_uncertain_set = np.zeros(len(new_theta), dtype=bool)

    for i, (new_theta_i, true_solution, safe_solution) in enumerate(
        zip(new_theta, true_solutions, safe_solutions)
    ):
        safe_reward[i] = np.dot(safe_solution, new_theta_i)
        true_reward[i] = np.dot(true_solution, new_theta_i)

        safe_constraint_violations[i] = np.sum(
            np.maximum(phi @ safe_solution - threshold, 0)
        )

        if method == "constraint_learning":
            true_solution_in_safe_set[i] = algorithm.is_in_polytope(
                true_solution, safe_polytope, inclusive=True, eps=TOLERANCE
            )
            true_solution_in_unsafe_set[i] = False
            for unsafe_polytope in unsafe_polytopes:
                if algorithm.is_in_polytope(
                    true_solution, unsafe_polytope, inclusive=False, eps=TOLERANCE
                ):
                    true_solution_in_unsafe_set[i] = True
                    break

            true_solution_in_uncertain_set[i] = not (
                true_solution_in_safe_set[i] or true_solution_in_unsafe_set[i]
            )

    return results.CEHighwayExperimentResult(
        true_reward=true_reward,
        safe_reward=safe_reward,
        found_safe_solution=np.array(found_safe_solution),
        safe_constraint_violations=safe_constraint_violations,
        true_solution_in_safe_set=true_solution_in_safe_set,
        true_solution_in_unsafe_set=true_solution_in_unsafe_set,
        true_solution_in_uncertain_set=true_solution_in_uncertain_set,
    )
