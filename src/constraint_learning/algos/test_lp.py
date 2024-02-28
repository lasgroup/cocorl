import einops
import numpy as np
import pytest

from constraint_learning.algos import lp
from constraint_learning.envs import tabular


@pytest.fixture(scope="module", params=[False, True])
def simple_cmdp1(request):
    # Rewards for each state (state 0 and state 1)
    rewards = np.array([0, 2])
    # Transition probabilities for each action and state pair:
    # Action 0 transitions from state 0 to state 0 and from state 1 to state 1.
    # Action 1 transitions from state 0 to state 1 and from state 1 to state 0.
    transitions = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    discount_factor = 0.5
    return tabular.TabularCMDP(
        rewards,
        transitions,
        discount_factor=discount_factor,
        use_sparse_transitions=request.param,
    )


@pytest.fixture(scope="module", params=[False, True])
def simple_cmdp2(request):
    # Rewards for each state (state 0 and state 1)
    rewards = np.array([0, 1])
    # Transition probabilities for each action and state pair:
    # Action 0 transitions with 0.5 probability from state 0 to state 0 or state 1,
    #   and from state 1 to state 1.
    # Action 1 transitions from state 0 to state 1,
    #   and with 0.5 probability from state 1 to state 0 or state 1.
    transitions = np.array([[[0.5, 0.5], [0, 1]], [[0, 1], [0.5, 0.5]]])
    discount_factor = 0.5
    return tabular.TabularCMDP(
        rewards,
        transitions,
        discount_factor=discount_factor,
        use_sparse_transitions=request.param,
    )


@pytest.fixture(scope="module", params=[False, True])
def simple_cmdp3(request):
    # Rewards for each state (state 0 and state 1)
    rewards = np.array([0, 2])
    # Costs for each state (state 0 and state 1)
    costs = np.array([[0, 1], [0, 0]])
    cost_limits = np.array([0, 0])
    # Initial state distribution
    initial_state_distribution = np.array([0.5, 0.5])
    # Transition probabilities for each action and state pair:
    # Action 0 transitions from state 0 to state 0 and from state 1 to state 1.
    # Action 1 transitions from state 0 to state 1 and from state 1 to state 0.
    transitions = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    discount_factor = 0.9
    return tabular.TabularCMDP(
        rewards,
        transitions,
        costs=costs,
        cost_limits=cost_limits,
        initial_state_distribution=initial_state_distribution,
        discount_factor=discount_factor,
        use_sparse_transitions=request.param,
    )


def test_lp_solution_simple_cmdp1(simple_cmdp1):
    expected_policy = np.array([[0, 1], [1, 0]])
    lp_solution = lp.TabularLPSolver(simple_cmdp1).solve()
    np.testing.assert_array_equal(lp_solution.policy, expected_policy)
    lp_solution = lp.TabularLPSolver(simple_cmdp1).solve(
        rewards=simple_cmdp1.rewards,
        costs=simple_cmdp1.costs,
        cost_limits=simple_cmdp1.cost_limits,
    )
    np.testing.assert_array_equal(lp_solution.policy, expected_policy)


def test_lp_solution_simple_cmdp2(simple_cmdp2):
    expected_policy = np.array([[0, 1], [1, 0]])
    lp_solution = lp.TabularLPSolver(simple_cmdp2).solve()
    np.testing.assert_array_equal(lp_solution.policy, expected_policy)
    lp_solution = lp.TabularLPSolver(simple_cmdp2).solve(
        rewards=simple_cmdp2.rewards,
        costs=simple_cmdp2.costs,
        cost_limits=simple_cmdp2.cost_limits,
    )
    np.testing.assert_array_equal(lp_solution.policy, expected_policy)


def test_lp_solution_simple_cmdp3(simple_cmdp3):
    expected_policy = np.array([[1, 0], [0, 1]])
    lp_solution = lp.TabularLPSolver(simple_cmdp3).solve()
    np.testing.assert_array_equal(lp_solution.policy, expected_policy)
    lp_solution = lp.TabularLPSolver(simple_cmdp3).solve(
        rewards=simple_cmdp3.rewards,
        costs=simple_cmdp3.costs,
        cost_limits=simple_cmdp3.cost_limits,
    )
    np.testing.assert_array_equal(lp_solution.policy, expected_policy)


LEFT, RIGHT, UP, DOWN, STAY = (
    tabular.Gridworld.LEFT,
    tabular.Gridworld.RIGHT,
    tabular.Gridworld.UP,
    tabular.Gridworld.DOWN,
    tabular.Gridworld.STAY,
)

# Gridworld coordinates (x, y):
#  (0, 0)  (1, 0)  (2, 0)
#  (0, 1)  (1, 1)  (2, 1)
#  (0, 2)  (1, 2)  (2, 2)


@pytest.mark.parametrize("use_sparse_transitions", [False])
@pytest.mark.parametrize("random_action_prob", [0, 0.2])
@pytest.mark.parametrize("num_forbidden", [0, 1, 2])
@pytest.mark.parametrize("env_seed", [1, 2, 3])
def test_gridworld_lp_solution_goes_to_goal_and_avoids_forbidden(
    use_sparse_transitions, random_action_prob, num_forbidden, env_seed
):
    # basic consistency checks of the LP solution in simple gridworld environments
    # many of the checks rely on a single goal and at most one constraint

    gridworld = tabular.Gridworld(
        width=3,
        height=3,
        num_goals=1,
        num_forbidden=num_forbidden,
        num_constraints=1,
        discount_factor=0.9,
        env_seed=env_seed,
        random_action_prob=random_action_prob,
        use_sparse_transitions=use_sparse_transitions,
    )
    lp_solution = lp.TabularLPSolver(gridworld).solve()

    goal_state = gridworld.goal_states[0]

    # Check that policy maximizes probability of staying at the goal
    if random_action_prob == 0 or num_forbidden == 0:
        # for stochastic transitions with forbidden states, it can be optimal to
        # move away from the goal to avoid the forbidden state
        goal_action = lp_solution.policy[goal_state, :].argmax()
        assert (
            gridworld.transitions[:, goal_state, goal_state].max()
            == gridworld.transitions[goal_action, goal_state, goal_state]
        )

    # Check that policy goes to goal
    state_occupancy = lp_solution.occupancy.sum(axis=1)
    assert state_occupancy[goal_state] > 0
    if random_action_prob == 0 or num_forbidden == 0:
        assert state_occupancy.argmax() == goal_state

    # Check that policy avoids forbidden states
    if num_forbidden == 1:
        forbidden_state = gridworld.constraint_states[0][0]

        # should be more often at goal then in forbidden state
        assert (
            lp_solution.occupancy[forbidden_state].sum()
            < lp_solution.occupancy[goal_state].sum()
        )

        for state in range(gridworld.num_states):
            if state != forbidden_state:
                # action should not maximize probability of going to forbidden state
                action = lp_solution.policy[state, :].argmax()
                min_prob = gridworld.transitions[:, state, forbidden_state].max()
                max_prob = gridworld.transitions[:, state, forbidden_state].max()
                if min_prob < max_prob:
                    assert (
                        gridworld.transitions[action, state, forbidden_state] < max_prob
                    )


@pytest.mark.parametrize("use_sparse_transitions", [False])
@pytest.mark.parametrize("random_action_prob", [0, 0.2])
@pytest.mark.parametrize("num_forbidden", [1, 3, 5])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_gridworld_lp_solution_satisfies_constraints_on_taget_state_occupancy(
    use_sparse_transitions, random_action_prob, num_forbidden, seed
):
    # checks that the LP solution satisfies the constraints on the target state

    gridworld = tabular.Gridworld(
        width=3,
        height=3,
        num_goals=2,
        num_forbidden=num_forbidden,
        num_constraints=2,
        discount_factor=0.9,
        env_seed=seed,
        random_action_prob=random_action_prob,
        use_sparse_transitions=use_sparse_transitions,
    )

    np.random.seed(seed)
    rewards = np.random.normal(0, 1, size=(gridworld.num_states,))
    costs = np.random.normal(
        0, 1, size=(gridworld.num_constraints, gridworld.num_states)
    )
    cost_limits = np.random.uniform(0.2, 1, size=(gridworld.num_constraints,))

    lp_solution = lp.TabularLPSolver(gridworld).solve(
        rewards=rewards, costs=costs, cost_limits=cost_limits, tolerance=1e-6
    )

    target_state_occupancy = einops.einsum(
        lp_solution.occupancy, gridworld.transitions, "s a, a s next_s -> next_s"
    )
    np.testing.assert_array_less(costs @ target_state_occupancy, cost_limits + 1e-6)


@pytest.mark.parametrize("use_sparse_transitions", [False])
@pytest.mark.parametrize("random_action_prob", [0, 0.2])
@pytest.mark.parametrize("num_forbidden", [1, 3, 5])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_gridworld_lp_solution_is_deterministic(
    use_sparse_transitions, random_action_prob, num_forbidden, seed
):
    # check that the LP solution is the same between runs

    gridworld = tabular.Gridworld(
        width=3,
        height=3,
        num_goals=2,
        num_forbidden=num_forbidden,
        num_constraints=2,
        discount_factor=0.9,
        env_seed=seed,
        random_action_prob=random_action_prob,
        use_sparse_transitions=use_sparse_transitions,
    )

    np.random.seed(seed)
    rewards = np.random.normal(0, 1, size=(gridworld.num_states,))
    costs = np.random.normal(
        0, 1, size=(gridworld.num_constraints, gridworld.num_states)
    )
    cost_limits = np.random.uniform(0.2, 1, size=(gridworld.num_constraints,))

    lp_solution1 = lp.TabularLPSolver(gridworld).solve(
        rewards=rewards, costs=costs, cost_limits=cost_limits
    )

    lp_solution2 = lp.TabularLPSolver(gridworld).solve(
        rewards=rewards, costs=costs, cost_limits=cost_limits
    )

    np.testing.assert_array_equal(lp_solution1.policy, lp_solution2.policy)
    np.testing.assert_allclose(lp_solution1.occupancy, lp_solution2.occupancy)
    np.testing.assert_allclose(lp_solution1.reward, lp_solution2.reward)
    np.testing.assert_allclose(lp_solution1.costs, lp_solution2.costs)


@pytest.mark.parametrize("use_sparse_transitions", [False, True])
def test_infeasible_environment_raises_value_error(use_sparse_transitions):
    gridworld = tabular.Gridworld(
        width=3,
        height=3,
        num_goals=0,
        num_forbidden=9,
        num_constraints=1,
        discount_factor=0.9,
        ensure_feasibility=False,
        use_sparse_transitions=use_sparse_transitions,
    )
    gridworld.cost_limits = np.array([0])

    with pytest.raises(ValueError, match="Could not solve LP.*"):
        lp.TabularLPSolver(gridworld).solve()
