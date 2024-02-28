import einops
import numpy as np
import pytest

from constraint_learning.algos import lp
from constraint_learning.envs import tabular
from constraint_learning.linear.irl import max_ent_tabular


def _get_gridworld(seed):
    return tabular.Gridworld(
        width=3,
        height=3,
        num_goals=1,
        num_forbidden=1,
        num_constraints=1,
        discount_factor=0.97,
        random_action_prob=0.2,
        ensure_feasibility=True,
        use_sparse_transitions=False,
        env_seed=seed,
    )


def _get_random_mdp(seed):
    num_states = 10
    num_actions = 4
    np.random.seed(seed)
    transitions = np.random.random((num_actions, num_states, num_states))
    transitions = transitions / transitions.sum(axis=2, keepdims=True)
    init_state_dist = np.random.random((num_states,))
    init_state_dist = init_state_dist / init_state_dist.sum()
    reward = np.random.random((num_states))
    return tabular.TabularCMDP(reward, transitions, discount_factor=0.9)


@pytest.mark.parametrize("get_env", [_get_random_mdp, _get_gridworld])
@pytest.mark.parametrize(
    "constraint,known_reward", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_max_ent_irl_infers_correct_reward_in_single_policy(
    get_env, constraint, known_reward, seed
):
    mdp = get_env(seed)

    expert_policy = lp.TabularLPSolver(mdp).solve(no_constraints=True).policy

    irl = max_ent_tabular.TabularMaxEntIRL(
        mdp,
        expert_policy,
        learning_rate=1,
        max_iter=100,
        beta=1,
        regularizer=0.0001,
        convergence_threshold=1e-3,
        shared_constraint=constraint,
        known_rewards=np.zeros((1, mdp.num_states)) if known_reward else None,
    )

    if constraint:
        inferred_reward, inferred_constraint = irl.run(verbose=False)
        if known_reward:
            np.testing.assert_allclose(inferred_reward, 0)
        inferred_reward = inferred_reward + inferred_constraint
    else:
        inferred_reward = irl.run(verbose=False)

    inferred_policy = (
        lp.TabularLPSolver(mdp)
        .solve(rewards=inferred_reward, no_constraints=True)
        .policy
    )

    # compare transitions because policies can be different but still lead to the same
    # transitions (e.g., in gridworld when walking against a wall)

    expert_transition = einops.einsum(
        expert_policy, mdp.transitions, "s a, a s next_s -> s next_s"
    )
    inferred_transition = einops.einsum(
        inferred_policy, mdp.transitions, "s a, a s next_s -> s next_s"
    )
    np.testing.assert_allclose(expert_transition, inferred_transition)


@pytest.mark.parametrize("get_env", [_get_random_mdp, _get_gridworld])
@pytest.mark.parametrize("known_reward", [False, True])
@pytest.mark.parametrize("num_policy", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_max_ent_irl_infers_correct_reward_in_multi_policy(
    get_env, known_reward, num_policy, seed
):
    np.random.seed(seed)
    mdp = get_env(seed)

    true_rewards = np.random.uniform(low=0, high=1, size=(num_policy, mdp.num_states))
    true_constraint = np.random.uniform(low=0, high=1, size=(mdp.num_states))

    expert_policy = np.array(
        [
            lp.TabularLPSolver(mdp)
            .solve(rewards=r + true_constraint, no_constraints=True)
            .policy
            for r in true_rewards
        ]
    )

    irl = max_ent_tabular.TabularMaxEntIRL(
        mdp,
        expert_policy,
        learning_rate=1,
        max_iter=1000,
        beta=10,  # need higher beta to learn shared constraint
        regularizer=0,
        convergence_threshold=1e-4,
        shared_constraint=True,
        known_rewards=true_rewards if known_reward else None,
    )

    inferred_rewards, inferred_constraint = irl.run(verbose=False)

    inferred_policy = np.array(
        [
            (
                lp.TabularLPSolver(mdp)
                .solve(rewards=r + inferred_constraint, no_constraints=True)
                .policy
            )
            for r in inferred_rewards
        ]
    )

    # compare transitions because policies can be different but still lead to the same
    # transitions (e.g., in gridworld when walking against a wall)

    expert_transition = einops.einsum(
        expert_policy, mdp.transitions, "p s a, a s next_s -> p s next_s"
    )
    inferred_transition = einops.einsum(
        inferred_policy, mdp.transitions, "p s a, a s next_s -> p s next_s"
    )
    np.testing.assert_allclose(expert_transition, inferred_transition)
