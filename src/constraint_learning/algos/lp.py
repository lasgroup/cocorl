import dataclasses
import time
from typing import Optional

import einops
import numpy as np
from scipy.optimize import linprog


@dataclasses.dataclass
class TabularLPSolution:
    """LP Solution to a tabular CMDP.

    Attributes:
        policy: policy with shape (num_states, num_actions)
        occupancy: occupancy measure with shape (num_states, num_actions)
        reward: reward of the policy
        costs: costs of the policy
    """

    policy: np.ndarray
    occupancy: np.ndarray
    reward: float
    costs: np.ndarray

    from typing import Optional


class TabularLPSolver:
    def __init__(self, cmdp):
        self.cmdp = cmdp
        self._solve_time = 0.0

    def solve(
        self,
        rewards: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
        cost_limits: Optional[np.ndarray] = None,
        tolerance: float = 0.0,
        no_constraints: bool = False,
    ) -> TabularLPSolution:
        """Find optimal policy by linear programming.

        If rewards is None, the rewards of the CMDP are used.

        If costs is None, the costs of the CMDP (or no constraints) are used.

        Args:
            rewards: 1D-array of rewards
            costs: 2D-array of shape (num_constraints, num_states) containing the costs
            cost_limits: 1D-array of shape (num_constraints) containing the cost limits
            tolerance: safety margin for the cost limits
            no_constraints: Unconstrained optimization.
        """
        t = time.time()
        num_states, num_actions = self.cmdp.num_states, self.cmdp.num_actions
        transitions, discount_factor = self.cmdp.transitions, self.cmdp.discount_factor

        if rewards is None:
            rewards = self.cmdp.rewards

        if no_constraints:
            costs, cost_limits = None, None
        elif costs is None:
            costs = self.cmdp.costs
            cost_limits = self.cmdp.cost_limits

        # Variables: d(s, a), where d is the occupancy measure
        num_vars = num_states * num_actions

        # State-action rewards
        r_sa = einops.einsum(rewards, transitions, "next_s, a s next_s -> s a")
        c = -einops.rearrange(r_sa, "s a -> (s a) 1")

        # Constraints:
        # 1. Bellman condition on occupancy measure:
        # sum_a d(s, a) == mu(s) + gamma * sum_{s',a'} p(s | s', a') * d(s', a')
        A = np.kron(
            np.eye(num_states), np.ones(num_actions)
        ) - discount_factor * einops.rearrange(
            transitions, "a s next_s -> next_s (s a)"
        )
        b = self.cmdp.initial_state_distribution

        # 2. Positive occupancy measure: d(s, a) >= 0
        G = -np.eye(num_vars)
        h = np.zeros(num_vars)

        # 3. (optionally) Constraints: sum_{s,a} d(s, a) * c_i(s, a) <= \tau_i
        if costs is not None:
            assert cost_limits is not None

            # compute state-action constraints
            cost_sa = einops.einsum(costs, transitions, "c next_s, a s next_s -> c s a")
            G2 = einops.rearrange(cost_sa, "c s a -> c (s a)")
            h2 = cost_limits

            if tolerance > 0:
                # add uncertainty vars
                c = np.vstack([c, np.zeros(num_vars).reshape((-1, 1))])
                A = np.hstack([A, np.zeros((A.shape[0], num_vars))])
                G = np.hstack([G, np.zeros((G.shape[0], num_vars))])

                G2 = np.hstack([G2, tolerance * np.ones((G2.shape[0], num_vars))])

                G3 = np.block(
                    [
                        [np.eye(num_vars), -np.eye(num_vars)],
                        [-np.eye(num_vars), np.eye(num_vars)],
                    ]
                )
                h3 = np.zeros(2 * num_vars)

                G = np.vstack([G, G2, G3])
                h = np.concatenate([h, h2, h3])
            else:
                G = np.vstack([G, G2])
                h = np.concatenate([h, h2])

        # Solve LP
        # Inequality constraints: G @ x <= h
        # Equality constraints:  A @ x == b
        options = {
            "disp": False,
            "presolve": False,
        }  # , "cholesky": False, "sym_pos": False}
        res = linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=b, options=options)

        if not res.success:
            raise ValueError(
                "Could not solve LP. " f"Status: {res.status}. Message: {res.message}"
            )

        occupancy = res.x[:num_vars].reshape(num_states, num_actions)

        # Compute rewards and costs from occupancy
        policy_reward = einops.einsum(r_sa, occupancy, "s a, s a ->")
        if costs is not None:
            policy_costs = einops.einsum(cost_sa, occupancy, "c s a, s a -> c")
        else:
            policy_costs = None

        # Get policy from occupancy measure
        policy = occupancy / occupancy.sum(axis=1, keepdims=True)

        self._solve_time += time.time() - t
        return TabularLPSolution(
            policy=policy,
            occupancy=occupancy,
            reward=policy_reward,
            costs=policy_costs,
        )
