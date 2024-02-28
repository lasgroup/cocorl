import gymnasium
import pytest

from constraint_learning.envs.controller_env import LinearVehicleIntersectionWrapper
from constraint_learning.envs.highway_envs import TruncateOnlyIntersectionEnv


def rollout_until_truncation(env: gymnasium.Env, max_duration: int) -> int:
    env.reset()
    done, steps = False, 0
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
        if steps >= max_duration or done:
            break
    return steps


@pytest.mark.parametrize("duration", [1, 5], ids=["1", "5"])
@pytest.mark.parametrize("policy_frequency", [1, 2], ids=["1", "2"])
@pytest.mark.parametrize(
    "make_env",
    [
        lambda: TruncateOnlyIntersectionEnv(),
        lambda: gymnasium.make("Intersect-TruncateOnly-v0"),
        lambda: LinearVehicleIntersectionWrapper(TruncateOnlyIntersectionEnv()),
        lambda: gymnasium.make("IntersectDefensive-TruncateOnly-v0"),
        lambda: gymnasium.make("IntersectAggressive-TruncateOnly-v0"),
    ],
    ids=[
        "TruncateOnlyIntersectionEnv",
        "gym.make('Intersect-TruncateOnly-v0')",
        "LinearVehicleIntersectionWrapper(TruncateOnlyIntersectionEnv())",
        "IntersectDefensive-TruncateOnly-v0",
        "IntersectAggressive-TruncateOnly-v0",
    ],
)
def test_truncated_only_env_truncates_on_duration(duration, policy_frequency, make_env):
    """Tests that the environment truncates the episode at the specified duration."""
    # duration is the number of seconds the episode should last
    # policy_frequency is the number actions per second
    env = make_env()
    env.configure({"duration": duration, "policy_frequency": policy_frequency})
    episode_length = duration * policy_frequency

    for _ in range(5):
        steps = rollout_until_truncation(env, max_duration=episode_length + 1)
        assert steps == episode_length


@pytest.mark.parametrize(
    "env_str,vehicle_type",
    [
        ("IntersectAggressive-TruncateOnly-v0", "AggressiveVehicle"),
        ("IntersectDefensive-TruncateOnly-v0", "DefensiveVehicle"),
    ],
)
def test_vehicle_type(env_str, vehicle_type):
    env = gymnasium.make(env_str)
    env.reset()

    # Step through the environment to spawn other vehicles
    for _ in range(10):
        env.step(env.action_space.sample())

    other_vehicles = [v for v in env.road.vehicles if v != env.vehicle]
    assert all(vehicle_type in type(v).__name__ for v in other_vehicles)
