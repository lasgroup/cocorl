import numpy as np
import pytest

from constraint_learning.algos import cross_entropy
from constraint_learning.envs import controller_env, feature_wrapper

# Create a dummy environment configuration
_ENV_CONFIG = {
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "duration": 15,
}

_ENV_NAME = "Intersect-TruncateOnly-v0"
_ENV_GOAL = controller_env.IntersectionGoal.LEFT
_REWARD_PARAMETERS = np.array([10, 0, 0, 0.1, 0, 0, 0, 0, 0])
_CONSTRAINT_PARAMETERS = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
)
_CONSTRAINT_THRESHOLDS = np.array([0.05, 0.1, 0.2, 0.1])


def _check_features(features):
    """Check that the result features are valid."""
    assert features is not None
    assert features.shape == (9,)

    # Features should be all positive are greater than 0
    assert np.all(features >= 0)

    # Indicator features should be less or equal to 1
    assert np.all(features[:3] <= 1)  # goal indicators
    assert np.all(features[-4:] <= 1)  # constraint indicators


def _check_result_params(result):
    """Check that the result parameters are valid."""
    assert result.acceleration is not None
    assert result.steering is not None
    assert result.acceleration.shape == (3,)
    assert result.steering.shape == (2,)
    assert result.goal.shape == (3,)


def test_make_env():
    env = cross_entropy._make_env(_ENV_NAME, _ENV_CONFIG, _ENV_GOAL)
    assert env is not None
    assert isinstance(env, feature_wrapper.IntersectionFeatureWrapper)
    assert isinstance(env.env, controller_env.LinearVehicleIntersectionWrapper)
    assert env.env.GOAL == _ENV_GOAL
    for key in _ENV_CONFIG:
        assert key in env.env.config
        assert _ENV_CONFIG[key] == env.env.config[key]


def test_get_features_from_parameter():
    parameter = cross_entropy.STARTING_MEAN
    features = cross_entropy._get_features_from_parameter(
        parameter=parameter,
        num_trajectories=3,
        env_name=_ENV_NAME,
        env_config=_ENV_CONFIG,
    )
    _check_features(features)


@pytest.mark.parametrize("num_jobs", [1, 2])
def test_cross_entropy_solver_without_constraints(num_jobs):
    solver = cross_entropy.CrossEntropySolver(_ENV_NAME, _ENV_CONFIG, num_jobs=num_jobs)

    iteration_count = 0

    def callback(locals, globals):
        nonlocal iteration_count
        iteration_count += 1

    result = solver.solve(
        reward_parameters=_REWARD_PARAMETERS,
        constraint_parameters=None,
        constraint_thresholds=None,
        iterations=3,
        num_candidates=3,
        num_elite=2,
        num_trajectories=1,
        verbose=False,
        callback=callback,
    )

    assert result.feasible
    _check_result_params(result)
    _check_features(result.features)
    assert iteration_count == 3


@pytest.mark.parametrize("num_jobs", [1, 2])
def test_cross_entropy_solver_with_constraints(num_jobs):
    solver = cross_entropy.CrossEntropySolver(_ENV_NAME, _ENV_CONFIG)

    iteration_count = 0

    def callback(locals, globals):
        nonlocal iteration_count
        iteration_count += 1

    result = solver.solve(
        reward_parameters=_REWARD_PARAMETERS,
        constraint_parameters=_CONSTRAINT_PARAMETERS,
        constraint_thresholds=_CONSTRAINT_THRESHOLDS,
        iterations=2,
        num_candidates=3,
        num_elite=2,
        num_trajectories=1,
        verbose=False,
        callback=callback,
    )

    _check_result_params(result)
    _check_features(result.features)
    assert iteration_count == 2


@pytest.mark.parametrize(
    "reward_param, expected_goal_param",
    [
        ([2, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0]),
        ([0, 0, 10, 0, 0, 0, 0, 0, 0], [0, 0, 1]),
    ],
)
def test_setting_goal_parameters_from_reward_argmax(reward_param, expected_goal_param):
    solver = cross_entropy.CrossEntropySolver(_ENV_NAME, _ENV_CONFIG)
    reward_param = np.array(reward_param)

    result = solver.solve(
        reward_parameters=reward_param,
        constraint_parameters=None,
        constraint_thresholds=None,
        argmax_goal_param=True,
        iterations=1,
        num_candidates=1,
        num_elite=1,
        num_trajectories=1,
        verbose=False,
    )

    _check_result_params(result)
    _check_features(result.features)
    assert result.goal.tolist() == expected_goal_param


@pytest.mark.parametrize(
    "starting, low, high, expected",
    [
        (np.ones(8), [2, 3, 4, 5, 6, 7, 8, 9], None, [2, 3, 4, 5, 6, 7, 8, 9]),
        (np.ones(8), None, [0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]),
        (
            np.ones(8),
            [2, 3, 4, 1, 1, 1, 0, 0],
            [3, 3, 4, 2, 2, 2, 0, 0],
            [2, 3, 4, 1, 1, 1, 0, 0],
        ),
    ],
)
def test_parameter_are_clipped_correctly(starting, low, high, expected):
    solver = cross_entropy.CrossEntropySolver(_ENV_NAME, _ENV_CONFIG)

    starting_mean = np.ones(8)
    starting_std = np.zeros(8)
    parameter_low = np.array([2, 3, 4, 5, 6, 7, 8, 9])

    result = solver.solve(
        reward_parameters=_REWARD_PARAMETERS,
        constraint_parameters=None,
        constraint_thresholds=None,
        argmax_goal_param=False,
        starting_mean=starting_mean,
        starting_std=starting_std,
        parameter_low=parameter_low,
        parameter_high=None,
        iterations=1,
        num_candidates=1,
        num_elite=1,
        num_trajectories=1,
        verbose=False,
    )

    _check_result_params(result)
    _check_features(result.features)
    assert result.goal.tolist() == [2, 3, 4]
    assert result.acceleration.tolist() == [5, 6, 7]
    assert result.steering.tolist() == [8, 9]
