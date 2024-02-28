import json
import time
from pathlib import Path

import numpy as np
import sacred

from constraint_learning.algos import cross_entropy
from constraint_learning.envs import controller_env, feature_wrapper
from constraint_learning.util import logging

ce_train = sacred.Experiment("ce_train", ingredients=[])
ce_train.observers = [
    logging.SetID(),
    sacred.observers.FileStorageObserver("ce_train"),
]


@ce_train.named_config
def debug():
    iterations = 1
    num_candidates = 1
    num_elite = 1
    num_trajectories = 1
    num_jobs = 1


@ce_train.named_config
def ground_truth_reward():
    reward_mean = [0, 0, 0, 0.1, 0, 0, 0, 0, 0]
    reward_std = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    env_goal = "o1"
    write_demonstration = False


@ce_train.config
def config():
    env_name = "Intersect-TruncateOnly-v0"
    env_goal = "o1"
    reward_mean = [0, 0, 0, 0.1, -0.2, 0, 0, 0, 0]
    reward_std = [0, 0, 0, 0.1, 0.1, 0, 0, 0, 0]
    iterations = 30
    num_candidates = 100
    num_elite = 20
    num_trajectories = 1
    num_jobs = 1
    solver_reinit = 1
    constraint_sort_method = "num_violations"
    write_demonstration = True
    experiment_label = None


@ce_train.automain
def main(
    _run,
    experiment_label,
    env_name,
    env_goal,
    reward_mean,
    reward_std,
    iterations,
    num_candidates,
    num_elite,
    num_trajectories,
    num_jobs,
    solver_reinit,
    constraint_sort_method,
    write_demonstration,
):
    env_config = {
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "duration": 15,
    }
    solver = cross_entropy.CrossEntropySolver(
        env_name,
        env_config,
        num_jobs=num_jobs,
        constraint_sort_method=constraint_sort_method,
    )

    # Sample reward parameter from the specified distribution
    reward_mean, reward_std = np.array(reward_mean), np.array(reward_std)
    assert reward_mean.shape == reward_std.shape
    assert len(reward_mean.shape) == 1
    reward_parameters = reward_mean + np.random.randn(reward_mean.shape[0]) * reward_std
    goal_idx = [
        controller_env.IntersectionGoal.LEFT,
        controller_env.IntersectionGoal.MIDDLE,
        controller_env.IntersectionGoal.RIGHT,
    ].index(env_goal)
    reward_parameters[goal_idx] = 10.0

    constraint_parameters = np.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    constraint_thresholds = np.array(
        [
            0.2,  # speed_gt_limit
            0.2,  # too_close_to_front_vehicle
            0.05,  # collision
            0.1,  # not_on_street
        ]
    )

    def callback(locals, globals):
        mu = locals["mu"]
        std = locals["std"]
        mu_features = locals["mu_features"]
        optimal_features = locals["optimal_features"]

        optimal_rew = np.dot(optimal_features, reward_parameters)
        optimal_const = constraint_parameters @ optimal_features - constraint_thresholds

        mu_rew = np.dot(mu_features, reward_parameters)
        mu_const = reward_parameters @ mu_features - constraint_thresholds

        _run.log_scalar("optimal_rew", optimal_rew)
        for i, c in enumerate(optimal_const):
            _run.log_scalar(f"optimal_const_{i}", c)

        _run.log_scalar("mu_rew", mu_rew)
        for i, c in enumerate(mu_const):
            _run.log_scalar(f"mu_const_{i}", c)

        for i, p in enumerate(mu):
            _run.log_scalar(f"mu_{i}", p)
            _run.log_scalar(f"std_{i}", std[i])

        feature_names = feature_wrapper.IntersectionFeatureWrapper.FEATURE_NAMES
        for feature, value in zip(feature_names, optimal_features):
            _run.log_scalar(f"optimal_feature_{feature}", value)

        for feature, value in zip(feature_names, mu_features):
            _run.log_scalar(f"mu_feature_{feature}", value)

    ce_results = [
        solver.solve(
            reward_parameters=reward_parameters,
            constraint_parameters=constraint_parameters,
            constraint_thresholds=constraint_thresholds,
            iterations=iterations,
            num_candidates=num_candidates,
            num_elite=num_elite,
            num_trajectories=num_trajectories,
            verbose=True,
            callback=callback,
        )
        for _ in range(solver_reinit)
    ]

    feasible = False
    final_result = ce_results[0]
    for ce_result in ce_results:
        if ce_result.feasible:
            feasible = True
            if ce_result.reward > final_result.reward:
                final_result = ce_result

    result = {
        "env_name": env_name,
        "reward_parameters": reward_parameters.tolist(),
        "constraint_parameters": constraint_parameters.tolist(),
        "constraint_thresholds": constraint_thresholds.tolist(),
        "goal": final_result.goal.tolist(),
        "acceleration": final_result.acceleration.tolist(),
        "steering": final_result.steering.tolist(),
        "features": final_result.features.tolist(),
        "feasible": feasible,
    }

    # Save the reward parameter, resulting parameters, and features to a file
    if write_demonstration:
        demonstrations_dir = Path("demonstrations")
        demonstrations_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")

        with open(demonstrations_dir / f"{timestamp}.json", "w") as f:
            json.dump(result, f, indent=4)

    return result
