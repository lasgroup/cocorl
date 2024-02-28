import dataclasses
from typing import Optional, Tuple

import einops
import numpy as np
import scipy
from scipy.optimize import linprog

from constraint_learning.envs import tabular


@dataclasses.dataclass
class MaxMarginIRLResult:
    """Result of a Maximum Margin IRL problem."""

    theta_hat: np.ndarray
    xi: np.ndarray


def max_margin(
    feature_expectations: np.ndarray,
    optimal_idx: int,
    required_slack: float = 0.0,
) -> MaxMarginIRLResult:
    """Solves the Maximum Margin IRL problem over a set of discrete policies.

    Args:
        feature_expectations: An (num_policies, num_features) array containing the
            feature expectations for each policy.
        optimal_idx: The index of the policy that is optimal.
        required_slack: The minimum value that the slack variables must take.

    Returns:
        A tuple containing two NumPy arrays:
            - The optimal theta_hat values (reward function parameters).
            - The optimal xi values (slack variables).

    Raises:
        ValueError: If the linear programming problem could not be solved.
    """
    assert feature_expectations.ndim == 2
    num_policies, num_features = feature_expectations.shape
    num_constraints = num_policies - 1
    assert 0 <= optimal_idx < num_policies

    # Objective function coefficients
    c = np.zeros(num_features + num_constraints)
    c[-num_constraints:] = -1

    # Inequality constraints matrix (A_ub) and vector (b_ub)
    A_ub = np.zeros((num_constraints, num_features + num_constraints))
    b_ub = np.zeros(num_constraints)

    # For each policy that is not optimal, add a constraint that says this policy
    # must have lower reward than the optimal policy
    for j in range(num_policies):
        if j != optimal_idx:
            # we have one constraint less than policies as we skip the optimal policy
            # when creating constraints. Hence, need to subtract 1 from optimal_idx on
            constraint_idx = j if j <= optimal_idx else j - 1
            # difference in feature expectations to compute difference in returns
            A_ub[constraint_idx, :num_features] = (
                feature_expectations[j] - feature_expectations[optimal_idx]
            )
            # slack variable
            A_ub[constraint_idx, num_features + constraint_idx] = 1

    # Bounds on the variables
    bounds = [(-1, 1) for _ in range(num_features)] + [
        (required_slack, None) for _ in range(num_constraints)
    ]

    # Solve the linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.success:
        theta_hat = res.x[:num_features]
        xi = res.x[num_features:]
        return MaxMarginIRLResult(theta_hat=theta_hat, xi=xi)
    raise ValueError("The linear programming problem could not be solved.")


def max_margin_tabular(
    mdp: tabular.TabularCMDP,
    expert_policy: np.ndarray,
    regularizer: float = 0.0,
    weight_bounds: Tuple[float, float] = (-1, 1),
):
    num_states, num_actions = mdp.num_states, mdp.num_actions
    assert expert_policy.shape == (num_states, num_actions)

    # transitions (action, state, next_state)
    gamma, transitions = mdp.discount_factor, mdp.transitions

    # variables \xi, r
    num_slack_vars = num_states * num_actions

    # maximize \sum_i \xi
    c = np.concatenate([-np.ones(num_slack_vars), np.zeros(num_states)])

    # (state, next_state)
    expert_transitions = einops.einsum(
        transitions, expert_policy, "a s next_s, s a -> s next_s"
    )

    # expert policy state_visitations d(s' | s)  (initial_state, state)
    # (I - gamma * P)^{-1}
    expert_visitations = np.eye(num_states) - gamma * expert_transitions
    expert_visitations = np.linalg.inv(expert_visitations)
    expert_visitations = gamma * expert_visitations @ expert_transitions

    visitation_diff = einops.einsum(
        expert_transitions - transitions,
        np.eye(num_states) + expert_visitations,
        "a s next_s, next_s final_s -> s a final_s",
    )
    visitation_diff = einops.rearrange(visitation_diff, "s a final_s -> (s a) final_s")

    A_ub = np.hstack([np.eye(num_slack_vars), -visitation_diff])
    b_ub = np.zeros(num_slack_vars)
    bounds = [(0, None)] * num_slack_vars + [weight_bounds] * num_states

    if regularizer > 0:
        # minimize -\sum_i \xi + \sum_i L_i
        # where L_i = |r_i|
        c = np.concatenate([c, regularizer * np.ones(num_states)])
        A_ub = np.block(
            [
                [A_ub, np.zeros((A_ub.shape[0], num_states))],
                # r <= L
                [
                    np.zeros((num_states, num_slack_vars)),
                    np.eye(num_states),
                    -np.eye(num_states),
                ],
                # -r <= L
                [
                    np.zeros((num_states, num_slack_vars)),
                    -np.eye(num_states),
                    -np.eye(num_states),
                ],
            ]
        )
        b_ub = np.concatenate([b_ub, np.zeros(num_states), np.zeros(num_states)])
        bounds = bounds + [(0, None)] * num_states

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if not res.success:
        raise ValueError("The linear programming IRL problem could not be solved.")

    if regularizer > 0:
        print("regularizer", res.x[-num_states:])

    xi, r = res.x[:num_slack_vars], res.x[num_slack_vars : num_slack_vars + num_states]
    return r, xi


def max_margin_tabular_multi_policy(
    mdp: tabular.TabularCMDP,
    expert_policy: np.ndarray,
    known_rewards: Optional[np.ndarray] = None,
    regularizer: float = 0.0,
    weight_bounds: Tuple[float, float] = (-1, 1),
):
    num_policies, num_states, num_actions = expert_policy.shape
    assert (mdp.num_states, num_actions) == (num_states, num_actions)

    if known_rewards is not None:
        assert known_rewards.shape == (num_policies, num_states)

    # transitions (action, state, next_state)
    gamma, transitions = mdp.discount_factor, mdp.transitions

    # variables \xi, r
    num_slack_vars = num_policies * num_states * num_actions

    # maximize \sum_i \xi
    if known_rewards is None:
        c = np.concatenate(
            [
                -np.ones(num_slack_vars),
                np.zeros(num_states),
                np.zeros(num_policies * num_states),
            ]
        )
    else:
        c = np.concatenate(
            [
                -np.ones(num_slack_vars),
                np.zeros(num_states),
                100 * np.ones(num_slack_vars),
            ]
        )

    # (state, next_state)
    expert_transitions = einops.einsum(
        transitions, expert_policy, "a s next_s, policy s a -> policy s next_s"
    )

    # expert policy state_visitations d(s' | s)  (initial_state, state)
    # (I - gamma * P)^{-1}
    expert_visitations = np.eye(num_states)[None, ...] - gamma * expert_transitions
    expert_visitations = np.linalg.inv(expert_visitations)
    expert_visitations = einops.einsum(
        gamma * expert_visitations,
        expert_transitions,
        "policy s next_s, policy next_s final_s -> policy s final_s",
    )

    visitation_diff = expert_transitions[:, None, ...] - transitions[None, ...]
    expert_visitations_plus_1 = np.eye(num_states)[None, ...] + expert_visitations
    visitation_diff = einops.einsum(
        visitation_diff,
        expert_visitations_plus_1,
        "policy a s next_s, policy next_s final_s -> policy s a final_s",
    )
    visitation_diff_flat = einops.rearrange(
        visitation_diff, "policy s a final_s -> (policy s a) final_s"
    )

    if known_rewards is None:
        visitation_diff_block = [
            einops.rearrange(visitation_diff_policy, "s a final_s -> (s a) final_s")
            for visitation_diff_policy in visitation_diff
        ]
        visitation_diff_block = scipy.linalg.block_diag(*visitation_diff_block)
        A_ub = np.hstack(
            [np.eye(num_slack_vars), -visitation_diff_flat, -visitation_diff_block]
        )
        b_ub = np.zeros(num_slack_vars)
        bounds = [(0, None)] * num_slack_vars + [weight_bounds] * (
            num_states + num_policies * num_states
        )

    else:
        A_ub = np.hstack(
            [np.eye(num_slack_vars), -visitation_diff_flat, -np.eye(num_slack_vars)]
        )
        b_ub = einops.einsum(
            visitation_diff,
            known_rewards,
            "policy s a final_s, policy final_s -> policy s a",
        )
        b_ub = einops.rearrange(b_ub, "policy s a -> (policy s a)")
        bounds = (
            [(0, None)] * num_slack_vars
            + [weight_bounds] * num_states
            + [(0, None)] * num_slack_vars
        )

    if regularizer > 0:
        # minimize -\sum_i \xi + \sum_i L_i
        # where L_i = |r_i|
        num_rew_weights = len(c) - num_slack_vars
        c = np.concatenate([c, regularizer * np.ones(num_rew_weights)])
        A_ub = np.block(
            [
                [A_ub, np.zeros((A_ub.shape[0], num_rew_weights))],
                # r <= L
                [
                    np.zeros((num_rew_weights, num_slack_vars)),
                    np.eye(num_rew_weights),
                    -np.eye(num_rew_weights),
                ],
                # -r <= L
                [
                    np.zeros((num_rew_weights, num_slack_vars)),
                    -np.eye(num_rew_weights),
                    -np.eye(num_rew_weights),
                ],
            ]
        )
        b_ub = np.concatenate(
            [b_ub, np.zeros(num_rew_weights), np.zeros(num_rew_weights)]
        )
        bounds = bounds + [(0, None)] * num_rew_weights

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if not res.success:
        raise ValueError("The linear programming IRL problem could not be solved.")

    if regularizer > 0:
        print("regularizer", res.x[-num_states:])

    xi, c = res.x[:num_slack_vars], res.x[num_slack_vars : num_slack_vars + num_states]

    if known_rewards is None:
        r = res.x[
            num_slack_vars
            + num_states : num_slack_vars
            + num_states
            + num_policies * num_states
        ].reshape((num_policies, num_states))
        return (c, r), xi
    else:
        c_slack = res.x[-num_slack_vars:]
        return c, (xi, c_slack)
