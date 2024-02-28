from typing import Optional

import einops
import numpy as np

from constraint_learning.envs import tabular


class TabularMaxEntIRL(object):
    """Implementation of MaxEnt IRL [1].

    [1] Ziebart et al. "Maximum Entropy Inverse Reinforcement Learning." AAAI 2008.

    Attributes:
        mdp: MDP to learn reward function for
        expert_policy: expert policy to learn reward function for
        learning_rate: learning rate of gradient descent
        max_iter: maximum number of iterations of gradient descent
        beta: temperature parameter of softmax
        regularizer: regularization parameter
        convergence_threshold: threshold for convergence of gradient descent
    """

    def __init__(
        self,
        mdp: tabular.TabularCMDP,
        expert_policy: np.ndarray,
        learning_rate: float = 1,
        max_iter: int = 100,
        beta: float = 1,
        regularizer: float = 0.01,
        convergence_threshold: float = 0.01,
        shared_constraint: bool = False,
        known_rewards: Optional[np.ndarray] = None,
    ):
        self.mdp = mdp
        self.expert_policy = expert_policy

        # expert_polcicy: [num_policies, num_states, num_actions]
        if len(self.expert_policy.shape) == 2:
            self.expert_policy = self.expert_policy[None, ...]
        self.num_policies = self.expert_policy.shape[0]

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta = beta
        self.regularizer = regularizer
        self.convergence_threshold = convergence_threshold

        self.shared_constraint = shared_constraint
        self.known_rewards = known_rewards

        self.expert_visitations = self.compute_feature_expectations(self.expert_policy)

    def compute_feature_expectations(self, policy: np.ndarray) -> np.ndarray:
        # Compute expert visitations P^\pi (I + \gamma (I - gamma*P^\pi)^{-1} P^\pi)
        policy_transitions = einops.einsum(
            self.mdp.transitions, policy, "a s next_s, policy s a -> policy s next_s"
        )
        policy_visitations = (
            np.eye(self.mdp.num_states)[None, ...]
            - self.mdp.discount_factor * policy_transitions
        )
        policy_visitations = np.linalg.inv(policy_visitations)
        policy_visitations = einops.einsum(
            self.mdp.discount_factor * policy_visitations,
            policy_transitions,
            "policy s next_s, policy next_s final_s -> policy s final_s",
        )
        policy_visitations = np.eye(self.mdp.num_states)[None, ...] + policy_visitations
        return einops.einsum(
            self.mdp.initial_state_distribution,
            policy_transitions,
            policy_visitations,
            "s, policy s next_s, policy next_s final_s -> policy final_s",
        )

    def soft_value_iteration(
        self,
        reward: np.ndarray,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-2,
        verbose: bool = False,
    ) -> np.ndarray:
        Q = np.zeros((self.num_policies, self.mdp.num_states, self.mdp.num_actions))

        if verbose:
            print("Running soft value iteration:", end=" ")

        for i in range(max_iterations):
            pi = np.exp(self.beta * (Q - Q.max(axis=-1, keepdims=True)))
            pi /= pi.sum(axis=-1, keepdims=True)

            old_Q = Q
            Q = reward + self.mdp.discount_factor * einops.einsum(
                self.mdp.transitions,
                pi,
                old_Q,
                "a s next_s, policy next_s next_a, policy next_s next_a -> policy s a",
            )

            if verbose:
                print("SoftVI Iteration:", i)
                print("pi:", pi)
                print("old_Q:", old_Q)
                print("Q:", Q)

            if np.sum(np.square(Q - old_Q)) < convergence_threshold:
                if verbose:
                    print("Soft VI converged. Stopping.")
                break
        return pi

    def compute_expected_state_visitation_frequency(
        self, reward: np.ndarray, constraint: np.ndarray
    ) -> np.ndarray:
        if self.shared_constraint:
            reward = reward + constraint[None, ...]
        state_reward = einops.einsum(
            self.mdp.transitions, reward, "a s next_s, policy next_s -> policy s a"
        )
        pi = self.soft_value_iteration(state_reward, verbose=False)
        return self.compute_feature_expectations(pi)

    def run(self, verbose: bool = False) -> np.ndarray:
        # Weights initialization
        if self.known_rewards is None:
            reward = (
                np.ones((self.num_policies, self.mdp.num_states)) / self.mdp.num_states
            )
        else:
            reward = self.known_rewards

        constraint = np.zeros(self.mdp.num_states) / self.mdp.num_states

        # Gradient descent
        for i in range(self.max_iter):
            expected_vf = self.compute_expected_state_visitation_frequency(
                reward, constraint
            )
            gradient = self.expert_visitations - expected_vf
            gradient_magnitude = np.sum(np.square(gradient))

            if self.shared_constraint:
                constraint_gradient = (
                    gradient.mean(axis=0) - self.regularizer * constraint
                )
                constraint = constraint + self.learning_rate * constraint_gradient

            if self.known_rewards is None:
                reward_gradient = gradient - self.regularizer * reward
                reward = reward + self.learning_rate * reward_gradient

            if verbose:
                print("Iteration", i)
                print("Observed feature expectations:", self.expert_visitations)
                print("Expected feature expectations:", expected_vf)
                print("Gradient magnitude:", gradient_magnitude)

                print("reward", reward)
                print("constraint", constraint)
                print("gradient", gradient)

            if gradient_magnitude < self.convergence_threshold:
                if verbose:
                    print("Max Ent IRL converged. Stopping.")
                break

        if not self.shared_constraint:
            reward -= np.min(reward, axis=1, keepdims=True)
            m = np.abs(reward).max(axis=1, keepdims=True)
            reward = np.where(m > 0, reward / m, reward)

        if self.num_policies == 1:
            reward = reward[0]

        if self.shared_constraint:
            return reward, constraint

        return reward


if __name__ == "__main__":
    from constraint_learning.algos import lp

    num_states = 3
    num_actions = 2

    np.random.seed(3)

    transitions = np.random.random((num_actions, num_states, num_states))
    transitions = transitions / transitions.sum(axis=2, keepdims=True)
    init_state_dist = np.random.random((num_states,))
    init_state_dist = init_state_dist / init_state_dist.sum()
    reward = np.random.random((num_states))

    mdp = tabular.TabularCMDP(reward, transitions, discount_factor=0.9)

    expert_policy = lp.TabularLPSolver(mdp).solve().policy

    irl = TabularMaxEntIRL(
        mdp, expert_policy, learning_rate=1, max_iter=1000, beta=1, regularizer=0.01
    )
    inferred_reward = irl.run(verbose=True)

    inferred_policy = lp.TabularLPSolver(mdp).solve(rewards=inferred_reward).policy

    print("reward", reward)
    print("inferred reward", inferred_reward)
    print("expert policy", expert_policy)
    print("inferred policy", inferred_policy)

    print("Reward error", np.sum(np.square(reward, inferred_reward)))
    print("Policy error:", np.sum(expert_policy != inferred_policy))
