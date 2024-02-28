"""Constraing learning experiment using cross entropy solver on highway_env."""
from typing import Any, Dict, Optional

import einops
import numpy as np
import sacred

from constraint_learning.algos import lp
from constraint_learning.envs import tabular
from constraint_learning.linear import algorithm
from constraint_learning.linear.irl import max_ent_tabular, max_margin
from constraint_learning.util import logging, results, sampling

TOLERANCE = 1e-4


ex = sacred.Experiment("gridworld_experiment_lp")
ex.observers = [
    logging.SetID(),
    sacred.observers.FileStorageObserver("results/gridworld"),
]


def get_solution_features(
    solver: lp.TabularLPSolver,
    env: tabular.TabularCMDP,
    *,
    theta: Optional[np.ndarray] = None,
    phi: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    tolerance: float = 0,
    return_policy: bool = False,
    no_constraints: bool = False,
) -> np.ndarray:
    """Solve environment with lp solver and compute state occupancy features."""
    rewards = env.rewards if theta is None else theta

    if no_constraints:
        costs, cost_limits = None, None
    else:
        costs = env.costs if phi is None else phi
        cost_limits = env.cost_limits if thresholds is None else thresholds

    lp_solution = solver.solve(
        rewards=rewards,
        costs=costs,
        cost_limits=cost_limits,
        tolerance=tolerance,
        no_constraints=no_constraints,
    )
    features = einops.einsum(
        lp_solution.occupancy,
        env.transitions,
        "s a, a s next_s -> next_s",
    )
    if return_policy:
        return features, lp_solution.policy
    return features


@ex.named_config
def debug():
    env_kwargs = {
        "width": 3,
        "height": 3,
        "num_goals": 2,
        "num_forbidden": 3,
        "num_constraints": 2,
        "random_action_prob": 0.2,
        "discount_factor": 0.97,
        "ensure_feasibility": True,
        "use_sparse_transitions": False,
    }
    num_thetas = 34
    num_new_thetas = 1
    method = "constraint_learning"
    seed = 86038
    experiment_label = None


@ex.named_config
def debug2():
    env_kwargs = {
        "discount_factor": 0.9,
        "ensure_feasibility": True,
        "height": 3,
        "num_constraints": 2,
        "num_forbidden": 3,
        "num_goals": 2,
        "random_action_prob": 0.0,
        "use_sparse_transitions": False,
        "width": 3,
    }
    env_transfer = (False,)
    experiment_label = "gridworld_small_exp1"
    max_thetas = 100
    method = "constraint_learning"
    new_theta_seed = 98321
    num_new_thetas = 30
    num_points = 30
    num_thetas = 30
    seed = 86861
    task_transfer = False


@ex.config
def cfg():
    env_kwargs = {
        "width": 3,
        "height": 3,
        "num_goals": 2,
        "num_forbidden": 3,
        "num_constraints": 2,
        "random_action_prob": 0.0,
        "discount_factor": 0.9,
        "ensure_feasibility": True,
        "use_sparse_transitions": False,
    }
    num_thetas = 5
    num_new_thetas = 5
    num_points = 20
    method = "constraint_learning"
    max_thetas = 100
    new_theta_seed = None
    task_transfer = False
    env_transfer = False

    seed = 1
    experiment_label = None


@ex.automain
def gridworld_experiment(
    _run,
    env_kwargs: Dict[str, Any],
    num_thetas: int,
    num_new_thetas: int,
    num_points: int,
    method: str,
    max_thetas: int,
    new_theta_seed: Optional[int],
    task_transfer: bool,
    env_transfer: bool,
    seed: int,
    experiment_label: str,
) -> results.CEHighwayExperimentResult:
    print(f"Experiment {experiment_label} (Seed: {seed})")

    if num_thetas > max_thetas:
        print(
            "Warning, max_thetas too low. "
            f"num_thetas: {num_thetas}; max_thetas: {max_thetas}"
        )
        max_thetas = num_thetas

    env_kwargs = dict(
        env_seed=seed,
        **env_kwargs,
    )
    env = tabular.Gridworld(**env_kwargs)
    solver = lp.TabularLPSolver(env)

    # get demonstrations
    reward_mean = env.sample_reward()
    reward_std = 0.1 * np.ones_like(reward_mean)

    theta = np.random.normal(reward_mean, reward_std, size=(max_thetas, env.num_states))
    theta = theta[:num_thetas]

    with sampling.temp_seed(new_theta_seed):
        if task_transfer:
            new_theta = np.array([env.sample_reward() for _ in range(num_new_thetas)])
        else:
            new_theta = np.random.normal(
                reward_mean, reward_std, size=(num_new_thetas, env.num_states)
            )

    demonstration_policies, demonstrations = [], []
    for th in theta:
        features, policy = get_solution_features(
            solver, env, theta=th, return_policy=True
        )
        demonstration_policies.append(policy)
        demonstrations.append(features)
    demonstrations = np.array(demonstrations)

    true_solutions = np.array(
        [get_solution_features(solver, env, theta=th) for th in new_theta]
    )

    # get worst possible reward for normalizing regret
    min_reward = np.array(
        [th @ get_solution_features(solver, env, theta=-th) for th in new_theta]
    )

    _, unique_idx = np.unique(np.round(demonstrations, 6), axis=0, return_index=True)

    print("Thetas:")
    for th in theta:
        print(th)
    print("Demonstrations:")
    for d in demonstrations:
        print(d)
    print("Unique demonstrations:", unique_idx)

    # get safe set
    if method == "constraint_learning":
        safe_polytope, unsafe_polytopes = algorithm.get_safe_set(
            demonstrations,
            num_points=num_points,
            stopping_dist=1e-8,
            min_singular_value=1e-8,
            orthogonal_tolerance=1e-8,
        )

        demos_in_safe_set = [
            algorithm.is_in_polytope(d, safe_polytope, inclusive=True, eps=TOLERANCE)
            for d in demonstrations
        ]
        dist = [
            algorithm.find_closest_point_in_polytope(d, safe_polytope)
            for d in demonstrations
        ]
        print(dist)
        print(demos_in_safe_set)

        A, b = np.copy(safe_polytope.A), np.copy(safe_polytope.b)
        num_inferred_constraints = A.shape[0]

        if env_transfer:
            env.random_action_prob = 0
            env.transitions = env._get_transitions()

        # solve for new rewards under inferred constraints
        safe_solutions = []
        found_safe_solution = []
        for th in new_theta:
            try:
                safe_solutions.append(
                    get_solution_features(
                        solver, env, theta=th, phi=A, thresholds=b, tolerance=0
                    )
                )
                found_safe_solution.append(True)
            except ValueError:
                safe_solutions.append(np.zeros(env.num_states))
                found_safe_solution.append(False)
    else:
        num_inferred_constraints = 0
        found_safe_solution = []

        if method == "vanilla_irl_max_margin":
            inferred_rewards = []
            for expert_policy in demonstration_policies:
                r, _ = max_margin.max_margin_tabular(
                    env, expert_policy, weight_bounds=(-1, 1)
                )
                inferred_rewards.append(r)
            irl_reward = np.mean(inferred_rewards, axis=0)
        elif method == "known_reward_irl_max_margin":
            irl_reward, (_, c_slack) = max_margin.max_margin_tabular_multi_policy(
                env,
                np.array(demonstration_policies),
                known_rewards=theta,
                weight_bounds=(-1, 1),
            )
            # hack to track slack for margin IRL; should be a proper entry in result
            num_inferred_constraints = c_slack.sum()
        elif method == "shared_reward_irl_max_margin":
            (irl_reward, separate), _ = max_margin.max_margin_tabular_multi_policy(
                env, np.array(demonstration_policies), weight_bounds=(-1, 1)
            )
        elif method == "vanilla_irl_max_ent":
            inferred_rewards = []
            for expert_policy in demonstration_policies:
                max_ent_irl = max_ent_tabular.TabularMaxEntIRL(
                    env,
                    expert_policy,
                    learning_rate=1.0,
                    max_iter=1000,
                    beta=10,
                    regularizer=1e-5,
                    convergence_threshold=1e-4,
                )
                r = max_ent_irl.run(verbose=True)
                inferred_rewards.append(r)
            irl_reward = np.mean(inferred_rewards, axis=0)
        elif method == "known_reward_irl_max_ent":
            _, irl_reward = max_ent_tabular.TabularMaxEntIRL(
                env,
                np.array(demonstration_policies),
                learning_rate=1.0,
                max_iter=1000,
                beta=10,
                regularizer=1e-5,
                convergence_threshold=1e-4,
                shared_constraint=True,
                known_rewards=theta,
            ).run(verbose=True)
        elif method == "shared_reward_irl_max_ent":
            _, irl_reward = max_ent_tabular.TabularMaxEntIRL(
                env,
                np.array(demonstration_policies),
                learning_rate=1.0,
                max_iter=1000,
                beta=10,
                regularizer=1e-5,
                convergence_threshold=1e-4,
                shared_constraint=True,
                known_rewards=None,
            ).run(verbose=True)
        else:
            raise ValueError(f"Unknown method {method}")

        if env_transfer:
            env.random_action_prob = 0
            env.transitions = env._get_transitions()

        safe_solutions = []
        for th in new_theta:
            safe_solutions.append(
                get_solution_features(
                    solver,
                    env,
                    theta=th + irl_reward,
                    phi=None,
                    thresholds=None,
                    tolerance=0,
                    no_constraints=True,
                )
            )

    phi = env.costs
    threshold = env.cost_limits

    safe_reward = np.zeros(len(new_theta))
    true_reward = np.zeros(len(new_theta))
    safe_constraint_violations = np.zeros(len(new_theta))
    safe_constraint_distance = np.zeros(len(new_theta))
    safe_constraint_max = np.zeros(len(new_theta))
    safe_inferred_constraint_violations = np.zeros(len(new_theta))
    safe_inferred_constraint_distance = np.zeros(len(new_theta))
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

        safe_constraint_distance[i] = algorithm.find_closest_point_in_polytope(
            safe_solution, algorithm.Polytope(phi, threshold)
        )

        safe_constraint_max[i] = np.max(phi @ safe_solution - threshold)

        if method == "constraint_learning":
            safe_inferred_constraint_violations[i] = np.sum(
                np.maximum(A @ safe_solution - b, 0)
            )

            safe_inferred_constraint_distance[
                i
            ] = algorithm.find_closest_point_in_polytope(safe_solution, safe_polytope)

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

    print("Safe solutions:")
    for solution in safe_solutions:
        print(solution)

    return results.GridworldExperimentResult(
        true_reward=true_reward,
        safe_reward=safe_reward,
        min_reward=min_reward,
        safe_constraint_violations=safe_constraint_violations,
        safe_constraint_distance=safe_constraint_distance,
        safe_constraint_max=safe_constraint_max,
        safe_inferred_constraint_violations=safe_inferred_constraint_violations,
        safe_inferred_constraint_distance=safe_inferred_constraint_distance,
        true_solution_in_safe_set=true_solution_in_safe_set,
        true_solution_in_unsafe_set=true_solution_in_unsafe_set,
        true_solution_in_uncertain_set=true_solution_in_uncertain_set,
        found_safe_solution=np.array(found_safe_solution),
        num_inferred_constraints=num_inferred_constraints,
    )
