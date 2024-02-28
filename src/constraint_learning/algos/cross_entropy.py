import dataclasses
import functools
import multiprocessing
import warnings
from typing import Any, Callable, Dict, Optional

import gymnasium
import numpy as np

from constraint_learning.envs import controller_env, feature_wrapper

_ACCELERATION_MEAN = np.array([0.3, 0.3, 2.0])
_STEERING_MEAN = np.array([5.0, 8.333333333333334])
_ACCELERATION_RANGE = np.array([[0.15, 0.15, 1.0], [0.45, 0.45, 3.0]])
_STEERING_RANGE = np.array([[4.93, 6.83333333], [5.07, 9.83333333]])
_ACCELERATION_STD = 2 * (_ACCELERATION_RANGE[1, :] - _ACCELERATION_RANGE[0, :])
_STEERING_STD = 2 * (_STEERING_RANGE[1, :] - _STEERING_RANGE[0, :])
_GOAL_MEAN = np.array([1 / 3, 1 / 3, 1 / 3])
_GOAL_STD = np.array([1, 1, 1])

STARTING_MEAN = np.concatenate([_GOAL_MEAN, _ACCELERATION_MEAN, _STEERING_MEAN])
STARTING_STD = np.concatenate([_GOAL_STD, _ACCELERATION_STD, _STEERING_STD])

GOALS = [
    controller_env.IntersectionGoal.LEFT,
    controller_env.IntersectionGoal.MIDDLE,
    controller_env.IntersectionGoal.RIGHT,
]

EnvConfig = Dict[str, Any]
Callback = Callable[[Dict[str, Any], Dict[str, Any]], None]


@dataclasses.dataclass
class CrossEntropySolverResult:
    """Class to store results from CEM.

    Attributes:
        - feasible: whether the solution is feasible
        - steering: steering parameters of the solution controller
        - acceleration: acceleration parameters of the solution controller
        - features: features of the solution for computing linear reward/constraint
        - reward: reward of the solution
        - constraint: constraint function value of the solution (minus thresholds)
    """

    feasible: bool
    steering: np.ndarray
    acceleration: np.ndarray
    goal: np.ndarray
    features: np.ndarray
    reward: float
    constraint: np.ndarray


def _make_env(
    env_name: str, env_config: EnvConfig, env_goal: controller_env.IntersectionGoal
):
    """Makes an environment from config and goal."""
    env = gymnasium.make(env_name)
    env.configure(env_config)  # type: ignore
    env = controller_env.LinearVehicleIntersectionWrapper(env)
    env.set_goal(env_goal)  # type: ignore
    # We compute rewards and constraints directly from features, so here we can choose
    # a zero reward parameter.
    env = feature_wrapper.IntersectionFeatureWrapper(env, reward_parameters=np.zeros(9))
    return env


def _get_features_from_parameter(
    parameter: np.ndarray, num_trajectories: int, env_name: str, env_config: EnvConfig
):
    """Rolls out policy with given parameters and return feature vector."""
    goal_param, acceleration_param, steering_param = np.split(parameter, (3, 6))
    env_goal = GOALS[goal_param.argmax()]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*A Box observation space has an unconventional shape"
        )
        env = _make_env(env_name, env_config, env_goal)

    vehicle_params = controller_env.ControllerParameters(
        acceleration=acceleration_param,
        steering=steering_param,
    )
    env.set_parameters(vehicle_params)
    features = []
    for _ in range(num_trajectories):
        env.reset()
        done = False
        env.reset()
        while not done:
            _, _, done, _, info = env.step(env.ACTIONS_INDEXES["IDLE"])
            features.append(info["feature_vector"])
    return np.array(features).mean(axis=0)


class CrossEntropySolver:
    def __init__(
        self,
        env_name: str,
        env_config: EnvConfig,
        parameter_dim: int = 8,
        num_jobs: int = 1,
        constraint_sort_method: str = "num_violations",
    ):
        self.env_name = env_name
        self.env_config = env_config
        self.parameter_dim = parameter_dim
        self.num_jobs = num_jobs
        assert constraint_sort_method in ("num_violations", "total_violation")
        self.constraint_sort_method = constraint_sort_method

    def solve(
        self,
        reward_parameters: np.ndarray,
        constraint_parameters: Optional[np.ndarray],
        constraint_thresholds: Optional[np.ndarray],
        iterations: int,
        num_candidates: int = 10,
        num_elite: int = 5,
        num_trajectories: int = 3,
        starting_mean: np.ndarray = STARTING_MEAN,
        starting_std: np.ndarray = STARTING_STD,
        parameter_low: Optional[np.ndarray] = None,
        parameter_high: Optional[np.ndarray] = None,
        argmax_goal_param: bool = True,
        verbose: bool = False,
        callback: Optional[Callback] = None,
    ) -> CrossEntropySolverResult:
        """Implements a cross entropy method for (constrained) policy optimization.

        If constraint_parameters is given to define a constraint function, the ordering
        defined by [1] is used to determine elite policies. Otherwise an ordering by
        rewards is used, i.e., the algorithm defaults to a vanilla CE method.

        The high-level algorithm is as follows:
            1. Sample a set of candidate policies from a Gaussian distribution.
            2. Roll out each policy, get features, rewards, and constraint values.
            3. Sort first by constraint violations, then by rewards.
            4. Keep the top num_elite policies.
            5. Compute the mean and standard deviation of the top policies.
            6. Repeat from step 1 until convergence.

        [1] Wen, Min, and Ufuk Topcu. "Constrained cross-entropy method for safe
            reinforcement learning." IEEE Transactions on Automatic Control (2020).

        Args:
            reward_parameters: Parameters of linear reward function.
            constraint_parameters: Parameters of constraint function(s).
            constraint_thresholds: Thresholds for constraints.
            iterations: Number of iterations to run.
            num_candidates: Number of candidate policies to sample per iteration.
            num_elite: Number of elite policies to keep per iteration.
            num_trajectories: Number of trajectories for evaluating policies.
            starting_mean: Mean of initial policy distribution.
            starting_std: Standard deviation of initial policy distribution.
            parameter_low: Lower bound for policy parameters.
            parameter_high: Upper bound for policy parameters.
            argmax_goal_param: Use argmax of reward weights to set goal parameter.
            verbose: Whether to print debug information.
            callback: Callback function to call after each iteration.

        Returns:
            Tuple of optimal policy parameters, and features of the policy.
        """

        if verbose:
            print("reward_parameters", reward_parameters)
            print("constraint_parameters", constraint_parameters)

        if parameter_low is None:
            parameter_low = -np.inf * np.ones(self.parameter_dim)
        if parameter_high is None:
            parameter_high = np.inf * np.ones(self.parameter_dim)

        mu, std = starting_mean, starting_std

        # Optimizing over the goal parameters is more principled, but currently
        # doesn't work well. So, as a workaround, we simply set the goal parameter to
        # the argmax of the goal reward weights. This is fine in practice because we
        # our reward functions are always constructed with a single goal.
        if argmax_goal_param:
            goal_param = np.zeros(3)
            goal_param[reward_parameters[:3].argmax()] = 1.0
            mu[:3] = goal_param
            std[:3] = 0.0

        optimal_policy = mu
        optimal_features = _get_features_from_parameter(
            optimal_policy,
            num_trajectories=num_trajectories,
            env_name=self.env_name,
            env_config=self.env_config,
        )
        optimal_rew = -np.inf
        found_feasible = False

        for i in range(iterations):
            if verbose:
                print("iteration", i)

            parameters = mu + std * np.random.randn(num_candidates, self.parameter_dim)
            parameters = np.clip(parameters, parameter_low, parameter_high)

            get_features = functools.partial(
                _get_features_from_parameter,
                num_trajectories=num_trajectories,
                env_name=self.env_name,
                env_config=self.env_config,
            )

            # optionally parallelize gathering trajectories
            if self.num_jobs == 1:
                features_map = [get_features(param) for param in parameters]
            else:
                with multiprocessing.Pool(self.num_jobs) as p:
                    features_map = p.map(get_features, parameters)

            candidate_features = np.array(features_map)
            candidate_rewards = candidate_features @ reward_parameters
            if constraint_parameters is not None:
                candidate_constraint_violations = (
                    candidate_features @ constraint_parameters.T - constraint_thresholds
                )

            # boolean vector of feasible candidates
            if constraint_parameters is None or constraint_thresholds is None:
                feasible = np.ones(num_candidates, dtype=bool)
            else:
                feasible = np.all(candidate_constraint_violations <= 0, axis=-1)

            # indices of feasible candidates that are better than previous best
            better = np.argwhere((candidate_rewards > optimal_rew) & feasible).flatten()

            if len(better):
                idx = better[-1]
                optimal_policy = parameters[idx]
                optimal_rew = candidate_rewards[idx]
                optimal_features = candidate_features[idx]
                found_feasible = True

            if constraint_parameters is None:
                # unconstrained optimization
                idx = np.argsort(candidate_rewards)[::-1]
            else:
                # constrained optimization
                total_violation = np.sum(
                    np.maximum(candidate_constraint_violations, 0), axis=-1
                )

                if self.constraint_sort_method == "num_violations":
                    # sort first by number of violations then amount of violation
                    num_violations = np.sum(
                        candidate_constraint_violations > 0, axis=-1
                    )
                    idx = np.lexsort((total_violation, num_violations))
                elif self.constraint_sort_method == "total_violation":
                    # sort only by amount of violation
                    idx = np.lexsort((total_violation,))

                # if all elite are feasible, sort them descending by reward
                if feasible[num_elite - 1]:
                    idx = sorted(idx[:num_elite], key=lambda k: -candidate_rewards[k])

            # keep elite policies to compute mean and std
            elite = parameters[idx[:num_elite]]
            mu = elite.mean(axis=0)
            std = elite.std(axis=0)

            if callback is not None or verbose:
                # only for logging purposes
                mu_features = _get_features_from_parameter(
                    mu,
                    num_trajectories=num_trajectories,
                    env_name=self.env_name,
                    env_config=self.env_config,
                )

            if callback is not None:
                callback(locals(), globals())

            if verbose:
                print("mu", mu)
                print("std", std)
                print("mean policy features", mu_features)
                print("mean policy reward:", np.dot(mu_features, reward_parameters))
                if constraint_parameters is not None:
                    print(
                        "mean policy constraint",
                        constraint_parameters @ mu_features - constraint_thresholds,
                    )

        goal, acceleration, steering = np.split(np.array(optimal_policy), (3, 6))
        optimal_reward = np.dot(optimal_features, reward_parameters)

        if constraint_parameters is not None:
            optimal_constraint = (
                constraint_parameters @ optimal_features - constraint_thresholds
            )
        else:
            optimal_constraint = None

        if verbose:
            print()
            print("optimal policy", optimal_policy)
            print("optimal policy features", optimal_features)
            print("optimal policy reward", optimal_reward)
            if constraint_parameters is not None:
                print(
                    "optimal policy constraint",
                    optimal_constraint,
                )
            print()

        return CrossEntropySolverResult(
            feasible=found_feasible,
            steering=steering,
            acceleration=acceleration,
            goal=goal,
            features=optimal_features,
            reward=float(optimal_reward),
            constraint=optimal_constraint,
        )
