from typing import Optional

import numpy as np

from constraint_learning.algos import cross_entropy


def max_ent_irl(
    solver: cross_entropy.CrossEntropySolver,
    demonstration_features: np.array,
    num_iterations: int,
    lr: float = 1.0,
    num_rew_features: int = 5,
    num_constraint_features: int = 4,
    learn_constraint: bool = True,
    learn_reward: bool = False,
    known_rewards: Optional[np.ndarray] = None,
    regularizer: float = 0.0,
    ce_solver_kwargs: dict = {},
    callback=None,
):
    num_demos, num_dim = demonstration_features.shape
    assert num_dim == num_rew_features + num_constraint_features

    if known_rewards is None:
        reward_weights = np.zeros((num_demos, num_dim))
    else:
        reward_weights = known_rewards

    # Init constraint parameters
    constraint_weight = np.zeros(num_rew_features + num_constraint_features)

    for it in range(num_iterations):
        demo_i = it % num_demos
        expert_features = demonstration_features[demo_i]

        theta = reward_weights[demo_i] + constraint_weight

        ce_result = solver.solve(
            reward_parameters=theta,
            constraint_parameters=None,
            constraint_thresholds=None,
            **ce_solver_kwargs,
            callback=None,
        )
        features = ce_result.features

        grad = expert_features - features

        if learn_reward:
            rew_grad = np.copy(grad)
            # Don't update the constraint feature weights.
            rew_grad[-num_constraint_features:] = 0
            rew_grad -= regularizer * reward_weights[demo_i]
            reward_weights[demo_i] += lr * rew_grad

        if learn_constraint:
            const_grad = np.copy(grad)
            # Don't update the reward feature weights.
            const_grad[:num_rew_features] = 0
            const_grad -= regularizer * constraint_weight
            constraint_weight += lr * const_grad

        if callback is not None:
            callback(locals(), globals())

    return reward_weights, constraint_weight
