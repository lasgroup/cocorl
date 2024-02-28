import numpy as np
import pytest

from constraint_learning.algos import lp
from constraint_learning.envs import tabular

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


@pytest.mark.parametrize("random_action_prob", [0, 0.2])
@pytest.mark.parametrize(
    "x1, y1, action, x2, y2",
    [
        (0, 0, STAY, 0, 0),
        (0, 0, RIGHT, 1, 0),
        (0, 0, DOWN, 0, 1),
        (1, 0, LEFT, 0, 0),
        (0, 1, UP, 0, 0),
        (2, 0, RIGHT, 2, 0),
        (0, 2, DOWN, 0, 2),
        (0, 0, UP, 0, 0),
        (0, 0, LEFT, 0, 0),
    ],
)
def test_gridworld_transitions(x1, y1, action, x2, y2, random_action_prob):
    gridworld = tabular.Gridworld(
        width=3,
        height=3,
        num_goals=1,
        num_forbidden=1,
        num_constraints=2,
        discount_factor=0.9,
        env_seed=1,
        random_action_prob=random_action_prob,
    )
    state = gridworld._get_state_from_xy(x1, y1)
    next_state = gridworld._get_state_from_xy(x2, y2)

    if random_action_prob == 0:
        assert gridworld.transitions[action, state, next_state] == 1
    else:
        # random action probability does not imply a probability for a specific
        # transition because it depends on reachable states, e.g. for a corner state
        # more of the random action probability causes staying in the same state
        assert gridworld.transitions[action, state, next_state] < 1

    np.testing.assert_almost_equal(gridworld.transitions[action, state, :].sum(), 1)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_env_seed_determines_tiles_rewards_and_costs(seed):
    kwargs = dict(
        width=3,
        height=3,
        num_goals=2,
        num_forbidden=2,
        num_constraints=2,
        discount_factor=0.9,
        env_seed=seed,
    )

    gridworld1 = tabular.Gridworld(**kwargs)
    gridworld2 = tabular.Gridworld(**kwargs)
    np.testing.assert_array_equal(gridworld1.tiles, gridworld2.tiles)
    np.testing.assert_array_equal(gridworld1.rewards, gridworld2.rewards)
    np.testing.assert_array_equal(gridworld1.costs, gridworld2.costs)
    np.testing.assert_allclose(gridworld1.cost_limits, gridworld2.cost_limits)


@pytest.mark.parametrize("num_goals", [0, 1, 2])
@pytest.mark.parametrize("num_forbidden", [0, 1, 2])
@pytest.mark.parametrize("num_constraints", [1, 2])
def test_tiles_reward_and_costs_consistency(num_goals, num_forbidden, num_constraints):
    width, height = 3, 3
    gridworld = tabular.Gridworld(
        width=width,
        height=height,
        num_goals=num_goals,
        num_forbidden=num_forbidden,
        num_constraints=num_constraints,
        discount_factor=0.9,
    )

    # Check shapes
    assert gridworld.rewards.shape == (width * height,)
    assert gridworld.costs.shape == (num_constraints, width * height)
    assert gridworld.cost_limits.shape == (num_constraints,)
    assert gridworld.tiles.shape == (height, width)

    assert np.all(gridworld.cost_limits.max() <= 1)
    assert np.all(gridworld.cost_limits.min() >= 0)
    assert np.all((gridworld.rewards == 0) | (gridworld.rewards == 1))
    assert np.all((gridworld.costs == 0) | (gridworld.costs == 1))

    # Check that rewards and costs are consistent with tiles
    for x in range(width):
        for y in range(height):
            tile = gridworld.tiles[y, x]
            state = gridworld._get_state_from_xy(x, y)
            if tile == 1:
                assert gridworld.rewards[state] == 1
            else:
                assert gridworld.rewards[state] == 0

            if tile >= 2:
                np.testing.assert_almost_equal(gridworld.costs[:, state].sum(), 1)
            else:
                np.testing.assert_almost_equal(gridworld.costs[:, state].sum(), 0)


@pytest.mark.parametrize("num_goals", [1, 2])
@pytest.mark.parametrize("num_forbidden", [1, 3])
@pytest.mark.parametrize("num_constraints", [1, 2])
@pytest.mark.parametrize("random_action_prob", [0, 0.5, 1.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gridworld_always_has_feasible_policy(
    num_goals, num_forbidden, num_constraints, random_action_prob, seed
):
    width, height = 3, 3
    gridworld = tabular.Gridworld(
        width=width,
        height=height,
        num_goals=num_goals,
        num_forbidden=num_forbidden,
        num_constraints=num_constraints,
        discount_factor=0.9,
        random_action_prob=random_action_prob,
        ensure_feasibility=True,
        env_seed=seed,
    )

    # Check that LP solver does not fail
    lp.TabularLPSolver(gridworld).solve()
