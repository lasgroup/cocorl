import einops
import numpy as np
import pytest

from constraint_learning.algos import lp
from constraint_learning.envs import tabular
from constraint_learning.linear.irl import max_margin


@pytest.mark.parametrize(
    "feature_expectations, optimal_idx",
    [
        (np.array([[1, 0], [0, 1]]), 0),
        (np.array([[2, 1], [1, 2], [0, 0]]), 2),
        (np.array([[1, 1], [2, 2], [3, 3]]), 1),
    ],
)
def test_max_margin(feature_expectations, optimal_idx):
    num_policies, num_features = feature_expectations.shape
    irl_result = max_margin.max_margin(feature_expectations, optimal_idx)

    for j in range(num_policies):
        if j != optimal_idx:
            constraint_idx = j if j <= optimal_idx else j - 1
            assert (
                irl_result.theta_hat.T
                @ (feature_expectations[optimal_idx] - feature_expectations[j])
                >= irl_result.xi[constraint_idx]
            )
            assert irl_result.xi[constraint_idx] >= 0


def test_max_margin_fails_when_there_is_no_solution():
    """Policy with features [1, 1] is never optimal if slack is strictly positive."""
    feature_expectations = np.array([[-1, -1], [1, 1], [2, 2]])
    optimal_idx = 1

    with pytest.raises(ValueError):
        max_margin.max_margin(feature_expectations, optimal_idx, required_slack=1e-6)


def _get_random_mdp(seed):
    np.random.seed(seed)
    # num_states, num_actions = 5, 4
    num_states, num_actions = 2, 2
    transitions = np.random.rand(num_actions, num_states, num_states)
    transitions /= transitions.sum(axis=-1, keepdims=True)
    rewards = np.random.rand(num_states)
    discount_factor = 0.8
    initial_state_distribution = np.ones(num_states) / num_states
    return tabular.TabularCMDP(
        rewards,
        transitions,
        discount_factor=discount_factor,
        initial_state_distribution=initial_state_distribution,
    )


def _get_gridworld(seed):
    return tabular.Gridworld(
        width=3,
        height=4,
        num_goals=2,
        num_forbidden=1,
        num_constraints=1,
        discount_factor=0.5,
        random_action_prob=0.1,
        ensure_feasibility=True,
        use_sparse_transitions=False,
        env_seed=seed,
    )


def _get_policy_visiations(mdp, policy):
    policy_transitions = einops.einsum(
        mdp.transitions, policy, "a s next_s, s a -> s next_s"
    )
    policy_visitations = (
        np.eye(mdp.num_states) - mdp.discount_factor * policy_transitions
    )
    policy_visitations = np.linalg.inv(policy_visitations)
    policy_visitations = mdp.initial_state_distribution @ policy_visitations
    policy_visitations = policy_visitations @ policy_transitions
    return policy_visitations


def _check_inferred_reward(r, xi, mdp):
    assert r is not None
    assert xi is not None
    assert r.shape == (mdp.num_states,)
    np.testing.assert_array_less(r, 1 + 1e-6)
    np.testing.assert_array_less(-r, 1 + 1e-6)
    np.testing.assert_array_less(-xi, 1e-6)
    assert not np.allclose(r, 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("get_env", [_get_random_mdp, _get_gridworld])
def test_max_margin_tabular_returns_reasonable_result(seed, get_env):
    mdp = get_env(seed=seed)
    expert_policy1 = np.random.rand(mdp.num_states, mdp.num_actions)
    expert_policy1 /= np.sum(expert_policy1, axis=1, keepdims=True)
    expert_policy2 = np.random.rand(mdp.num_states, mdp.num_actions)
    expert_policy2 /= np.sum(expert_policy2, axis=1, keepdims=True)

    r1, xi1 = max_margin.max_margin_tabular(mdp, expert_policy1)
    r2, xi2 = max_margin.max_margin_tabular(mdp, expert_policy2)

    _check_inferred_reward(r1, xi1, mdp)
    _check_inferred_reward(r2, xi2, mdp)

    expert1_visitations = _get_policy_visiations(mdp, expert_policy1)
    expert2_visitations = _get_policy_visiations(mdp, expert_policy2)
    assert expert1_visitations @ r1 > expert2_visitations @ r1 - 1e-6
    assert expert2_visitations @ r2 > expert1_visitations @ r2 - 1e-6


@pytest.mark.parametrize("seed", [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("regularizer", [0.0, 0.01])
@pytest.mark.parametrize("get_env", [_get_random_mdp, _get_gridworld])
def test_max_margin_tabular_recovers_correct_reward(seed, regularizer, get_env):
    mdp = get_env(seed=seed)

    expert_policy = lp.TabularLPSolver(mdp).solve().policy
    recovered_rewards, xi = max_margin.max_margin_tabular(
        mdp, expert_policy, regularizer=regularizer
    )
    recovered_policy = lp.TabularLPSolver(mdp).solve(rewards=recovered_rewards).policy
    expert_visitations = _get_policy_visiations(mdp, expert_policy)
    recovered_visitations = _get_policy_visiations(mdp, recovered_policy)

    _check_inferred_reward(recovered_rewards, xi, mdp)

    # Under the recovered reward, the expert should be optimal
    np.testing.assert_allclose(
        expert_visitations @ recovered_rewards,
        recovered_visitations @ recovered_rewards,
        atol=1e-6,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("regularizer", [0.0, 0.01])
@pytest.mark.parametrize("get_env", [_get_random_mdp, _get_gridworld])
@pytest.mark.parametrize("num_policies", [1, 2, 3])
@pytest.mark.parametrize("known_reward", [False, True])
def test_max_margin_tabular_multi_policy_reduces_to_irl(
    seed, regularizer, get_env, num_policies, known_reward
):
    mdp = get_env(seed=seed)

    expert_policy = lp.TabularLPSolver(mdp).solve().policy

    r1, xi = max_margin.max_margin_tabular(mdp, expert_policy, regularizer=regularizer)
    _check_inferred_reward(r1, xi, mdp)

    if known_reward:
        r2, (xi, c_slack) = max_margin.max_margin_tabular_multi_policy(
            mdp,
            np.array([expert_policy] * num_policies),
            known_rewards=np.zeros((num_policies, mdp.num_states)),
            regularizer=regularizer,
        )
        np.testing.assert_allclose(c_slack, 0)
        _check_inferred_reward(r2, xi, mdp)
        np.testing.assert_allclose(r1, r2, atol=1e-6)
    else:
        (c, r), xi = max_margin.max_margin_tabular_multi_policy(
            mdp,
            np.array([expert_policy] * num_policies),
            known_rewards=None,
            regularizer=regularizer,
        )
        # Check that the reward is the same for all policies
        for i in range(len(r) - 1):
            np.testing.assert_allclose(r[i + 1], r[0])
        r2 = r[0] + c
        r1 /= np.linalg.norm(r1)
        r2 /= np.linalg.norm(r2)
        np.testing.assert_allclose(r1, r2, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_max_margin_tabular_multi_policy_returns_reasonable_result(seed):
    mdp = _get_random_mdp(seed=seed)

    num_policies = 3

    expert_policy = np.random.uniform(
        0, 1, size=(num_policies, mdp.num_states, mdp.num_actions)
    )
    expert_policy /= np.sum(expert_policy, axis=-1, keepdims=True)

    (c, rewards), xi = max_margin.max_margin_tabular_multi_policy(mdp, expert_policy)
    _check_inferred_reward(c, xi, mdp)

    for expert, r in zip(expert_policy, rewards):
        _check_inferred_reward(r, xi, mdp)

        recovered_policy = lp.TabularLPSolver(mdp).solve(rewards=r + c).policy
        recovered_visitations = _get_policy_visiations(mdp, recovered_policy)
        expert_visitations = _get_policy_visiations(mdp, expert)

        # Expert should be optimal under reward r_i + c
        np.testing.assert_allclose(
            expert_visitations @ (r + c),
            recovered_visitations @ (r + c),
            atol=1e-6,
        )
