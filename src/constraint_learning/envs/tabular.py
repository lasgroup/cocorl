import contextlib
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from constraint_learning.algos import lp

EPS = np.finfo(np.float32).eps


class TabularCMDP:
    """Implements a simple tabular CMDP with discrete state and action spaces.

    States are labeled with integers and the rewards are a 1D-array.

    Rewards only depend on the target state of the agent, i.e. R(s, a, s') = R(s').

    The environment assumes infinite horizon, so the discount factor has to be < 1.

    Attributes:
        num_states: number of states
        num_actions: number of actions
        rewards: array of rewards of shape (num_states,)
        transitions: transition matrix with shape (num_actions, num_states, num_states)
        discount_factor: discount factor (gamma, should be 0 < gamma < 1)
        initial_state_distribution: probability of stariting in each state
        use_sparse_transitions: whether to use sparse matrix representations to speed up
            running and solving the environment
    """

    def __init__(
        self,
        rewards: np.ndarray,
        transitions: np.ndarray,
        costs: Optional[np.ndarray] = None,
        cost_limits: Optional[np.ndarray] = None,
        discount_factor: float = 0.99,
        initial_state_distribution: Optional[np.ndarray] = None,
        use_sparse_transitions: bool = False,
    ):
        assert len(transitions.shape) == 3
        self.num_actions, self.num_states, _ = transitions.shape

        assert rewards.shape == (self.num_states,)
        assert transitions.shape == (self.num_actions, self.num_states, self.num_states)
        assert (
            initial_state_distribution is None
            or initial_state_distribution.shape == (self.num_states,)
        )
        assert 0 < discount_factor < 1

        self.num_constraints = 0
        if costs is not None:
            assert cost_limits is not None
            self.num_constraints = cost_limits.shape[0]
            assert costs.shape == (self.num_constraints, self.num_states)
            assert cost_limits.shape == (self.num_constraints,)

        self.rewards = rewards
        self.costs = costs
        self.cost_limits = cost_limits
        self.transitions = transitions
        self.discount_factor = discount_factor

        if initial_state_distribution is None:
            initial_state_distribution = np.ones(self.num_states) / self.num_states
        self.initial_state_distribution = initial_state_distribution

        self.use_sparse_transitions = use_sparse_transitions
        self.sparse_transitions = None
        if self.use_sparse_transitions:
            self.sparse_transitions = [sp.csr_matrix(t) for t in self.transitions]

        self._is_deterministic = None


class Gridworld(TabularCMDP):
    """Gridworld."""

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4

    ACTION_STEP = {
        LEFT: (-1, 0),
        RIGHT: (1, 0),
        UP: (0, -1),
        DOWN: (0, 1),
        STAY: (0, 0),
    }

    def __init__(
        self,
        width: int,
        height: int,
        num_goals: int,
        num_forbidden: int,
        num_constraints: int,
        discount_factor: float,
        random_action_prob: float = 0,
        ensure_feasibility: bool = True,
        use_sparse_transitions: bool = False,
        env_seed: Optional[int] = None,
    ):
        assert height > 0
        assert width > 0
        assert 0 <= random_action_prob <= 1
        assert num_constraints >= 1

        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 5

        self.num_goals = num_goals
        self.num_forbidden = num_forbidden
        self.num_constraints = num_constraints
        self.random_action_prob = random_action_prob

        self._seed = env_seed or np.random.randint(0, 2 ** 32 - 1)
        transitions = self._get_transitions()

        if ensure_feasibility:
            feasible = False
            max_cost_limit = 1.0
            for it in range(10000):
                if (it + 1) % 100 == 0:
                    max_cost_limit *= 2
                self._make_tiles(env_seed)
                self._make_rewards_and_costs_from_tiles(max_cost_limit)

                env = TabularCMDP(
                    rewards=self.rewards,
                    transitions=transitions,
                    costs=self.costs,
                    cost_limits=self.cost_limits,
                    discount_factor=discount_factor,
                    initial_state_distribution=None,
                    use_sparse_transitions=use_sparse_transitions,
                )

                try:
                    lp.TabularLPSolver(env).solve()
                    feasible = True
                    print(
                        "Found feasible environment with max_cost_limit:",
                        max_cost_limit,
                    )
                    break
                except ValueError:
                    pass

            if not feasible:
                raise ValueError("Could not create feasible gridworld")
        else:
            self._make_tiles(env_seed)
            self._make_rewards_and_costs_from_tiles()

        super().__init__(
            rewards=self.rewards,
            transitions=transitions,
            costs=self.costs,
            cost_limits=self.cost_limits,
            discount_factor=discount_factor,
            initial_state_distribution=None,
            use_sparse_transitions=use_sparse_transitions,
        )

    @contextlib.contextmanager
    def _random_seed_context(self):
        orig_seed = np.random.randint(0, 2 ** 32 - 1)
        np.random.seed(self._seed)
        yield
        self._seed = np.random.randint(0, 2 ** 32 - 1)
        np.random.seed(orig_seed)

    def sample_reward(self):
        goal_count = 0
        rewards = np.zeros(self.num_states)
        while goal_count < self.num_goals:
            with self._random_seed_context():
                s = np.random.randint(0, self.num_states)
            x, y = self._get_xy_from_state(s)
            if self._constraint_active(x, y) == 0:
                with self._random_seed_context():
                    rewards[s] = np.random.uniform(0, 1)
                goal_count += 1
        return rewards

    def _make_tiles(self, seed) -> None:
        assert self.width * self.height >= self.num_goals + self.num_forbidden
        self.tiles = np.zeros((self.height, self.width), dtype=int)

        with self._random_seed_context():
            # place goal tiles
            for _ in range(self.num_goals):
                x, y = self._get_random_empty_tile()
                self.tiles[y, x] = 1

            # place forbidden tiles
            for _ in range(self.num_forbidden):
                x, y = self._get_random_empty_tile()
                self.tiles[y, x] = np.random.randint(2, 2 + self.num_constraints)

    def _is_goal(self, x: int, y: int) -> bool:
        return self.tiles[y, x] == 1

    def _constraint_active(self, x: int, y: int) -> int:
        return max(self.tiles[y, x] - 1, 0)

    def _make_rewards_and_costs_from_tiles(self, max_cost_limit: float = 1.0) -> None:
        self.rewards = np.zeros(self.height * self.width)
        self.costs = np.zeros((self.num_constraints, self.height * self.width))
        self.goal_states: List[int] = []
        self.constraint_states: List[List[int]] = [
            [] for _ in range(self.num_constraints)
        ]

        for y in range(self.height):
            for x in range(self.width):
                s = self._get_state_from_xy(x, y)
                if self._is_goal(x, y):
                    self.rewards[s] = 1
                    self.goal_states.append(s)
                c = self._constraint_active(x, y)
                if c > 0:
                    self.costs[c - 1, s] = 1
                    self.constraint_states[c - 1].append(s)

        num_states_per_constraint = np.array([len(s) for s in self.constraint_states])

        stay_policy = np.zeros((self.num_states, self.num_actions))
        stay_policy[:, self.STAY] = 1

        with self._random_seed_context():
            self.cost_limits = np.random.uniform(
                low=0,
                high=max_cost_limit,
                size=(self.num_constraints,),
            )

    def _get_random_empty_tile(self) -> Tuple[int, int]:
        """Returns position of a random empty tile (i.e., tile with value 0)."""
        assert np.any(self.tiles == 0)
        x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
        while self.tiles[y, x] != 0:
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
        return x, y

    def _get_transitions(self) -> np.ndarray:
        """Returns the full transition matrix for the gridworld.

        In particular, the dynamics are:
            - Actions UP, DOWN, LEFT, RIGHT always move the agent deterministically
              in the corresponding direction if it is not at the boundary of the
              world
            - If the agent would leave the grid with an action, the action has
              no effect, i.e. the agent stays in the same tile
            - The STAY action makes the agent stay in the same tile
            - With probability `random_action_prob`, the agent takes a random
              action instead of the intended action
        """
        num_states, num_actions = self.num_states, self.num_actions
        transitions = np.zeros((num_actions, num_states, num_states), dtype=np.double)
        for a in range(num_actions):
            for s1 in range(num_states):
                s2 = self._get_next_state_for_action(s1, a)
                transitions[a, s1, s2] = 1

        stochastic_transitions = np.zeros(
            (num_actions, num_states, num_states), dtype=np.double
        )
        for action_taken in range(num_actions):
            stochastic_transitions[action_taken] = (
                1 - self.random_action_prob
            ) * transitions[action_taken]
            for action_random in range(num_actions):
                stochastic_transitions[action_taken] += (
                    self.random_action_prob / num_actions * transitions[action_random]
                )

        return stochastic_transitions

    def _get_next_state_for_action(self, state, a):
        """Compute the next state given that action `a` is taken in state `state`."""
        x1, y1 = self._get_xy_from_state(state)
        diff = self.ACTION_STEP[a]
        x2, y2 = x1 + diff[0], y1 + diff[1]
        if self._transition_possible((x1, y1), (x2, y2)):
            return self._get_state_from_xy(x2, y2)
        return state

    def _within_grid(self, x, y):
        """Returns True iff the point (x, y) is a valid coordinate in the grid."""
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def _transition_possible(
        self, point1: Tuple[int, int], point2: Tuple[int, int]
    ) -> bool:
        """Returns True iff the transition from `point1` to `point2` is possible."""
        x1, y1 = point1
        x2, y2 = point2
        assert self._within_grid(x1, y1)
        if not self._within_grid(x2, y2):
            return False
        return True

    def _get_state_from_xy(self, x: int, y: int) -> int:
        """Returns the state index given an agent position x, y coordinates."""
        return x + y * self.width

    def _get_xy_from_state(self, state: int) -> Tuple[int, int]:
        """Returns the agent position given a state index."""
        x = int(state % self.width)
        y = int(state // self.width)
        return x, y
