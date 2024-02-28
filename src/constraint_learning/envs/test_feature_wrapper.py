import gymnasium
import highway_env  # noqa: F401
import numpy as np
import pytest

from constraint_learning.envs.feature_wrapper import IntersectionFeatureWrapper


@pytest.fixture
def intersection_env():
    env = gymnasium.make("intersection-v0")
    return env


@pytest.fixture
def wrapped_env(intersection_env):
    reward_parameters = np.ones(9)
    constraint_parameters = np.ones((3, 9))
    constraint_thresholds = np.ones(3)
    wrapped_env = IntersectionFeatureWrapper(
        intersection_env,
        reward_parameters,
        constraint_parameters,
        constraint_thresholds,
    )
    return wrapped_env


def test_step_output_shape(wrapped_env):
    state = wrapped_env.reset()
    action = wrapped_env.action_space.sample()
    next_state, reward, terminated, truncated, info = wrapped_env.step(action)

    assert isinstance(reward, float), "Reward is not a float"
    assert isinstance(terminated, bool), "Terminated is not a bool"
    assert isinstance(truncated, bool), "Truncated is not a bool"
    assert isinstance(info, dict), "Info is not a dict"


def test_info_contains_features(wrapped_env):
    state = wrapped_env.reset()
    action = wrapped_env.action_space.sample()
    _, _, _, _, info = wrapped_env.step(action)

    assert "features" in info, "info does not contain 'features'"
    assert "feature_vector" in info, "info does not contain 'feature_vector'"
    assert isinstance(info["features"], dict), "Features is not a dict"
    assert isinstance(
        info["feature_vector"], np.ndarray
    ), "Feature vector is not a numpy array"
    assert len(info["features"]) == 9, "missing feature"
    assert info["feature_vector"].shape == (
        9,
    ), "Feature vector is not the correct shape"


def test_modified_reward(wrapped_env):
    state = wrapped_env.reset()
    action = wrapped_env.action_space.sample()
    _, reward, _, _, info = wrapped_env.step(action)
    expected_reward = np.dot(wrapped_env.reward_parameters, info["feature_vector"])

    assert np.isclose(
        reward, expected_reward, atol=1e-8
    ), "Reward is not modified correctly"


def test_constraint_function(wrapped_env):
    state = wrapped_env.reset()
    action = wrapped_env.action_space.sample()
    _, _, _, _, info = wrapped_env.step(action)

    assert "cost" in info
    expected_constraint_function = (
        wrapped_env.constraint_parameters @ info["feature_vector"]
        - wrapped_env.constraint_thresholds
    )
    assert np.allclose(
        info["cost"], expected_constraint_function, atol=1e-8
    ), "Constraint function is not computed correctly"
    assert info["cost"].shape == (3,), "Constraint is not correct shape"


def test_no_constraints(intersection_env):
    reward_parameters = np.ones(9)
    wrapped_env_no_constraints = IntersectionFeatureWrapper(
        intersection_env, reward_parameters
    )

    state = wrapped_env_no_constraints.reset()
    action = wrapped_env_no_constraints.action_space.sample()
    _, _, _, _, info = wrapped_env_no_constraints.step(action)

    assert "constraint_function" not in info


def test_disable_modifying_reward(intersection_env):
    reward_parameters = np.ones(9)
    wrapped_env = IntersectionFeatureWrapper(
        intersection_env, reward_parameters, modify_reward=False
    )

    actions = [intersection_env.action_space.sample() for _ in range(5)]

    seed = 2
    intersection_env.reset(seed=seed)
    rewards = [intersection_env.step(a)[1] for a in actions]
    wrapped_env.reset(seed=seed)
    wrapped_rewards = [wrapped_env.step(a)[1] for a in actions]

    assert np.allclose(rewards, wrapped_rewards, atol=1e-8), (
        "Reward should not be modified if modify_reward=False. "
        "Original: {reward}, Modified: {wrapped_reward}"
    )
