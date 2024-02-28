# type: ignore
"""Some support functions for running highway envs with SB3."""
import copy

import numpy as np
from gymnasium.envs.registration import register
from gymnasium.utils import seeding
from highway_env import utils
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.vehicle.kinematics import Vehicle


class TruncateOnlyIntersectionEnv(IntersectionEnv):
    """An intersection env that terminates only on timeout (i.e. truncated).

    A typical Intersection environment terminates on car crashes or arrival
    at a goal, but here we remove these terminating conditions and an episode
    only ends when it has reached the maximum possible duration.
    This setting is suitable for reward learning.
    """

    def __init__(
        self,
        max_duration: int = 15,
        no_penalty_reward: bool = True,
        policy_frequency: float = 1,
        simulation_frequency: float = 15,
        **kwargs
    ):
        self.no_penalty_reward = no_penalty_reward
        super().__init__(**kwargs)
        self.config["duration"] = max_duration

        self.config["policy_frequency"] = policy_frequency
        self.config["simulation_frequency"] = simulation_frequency

    def _is_terminated(self) -> bool:
        return self.time >= self.config["duration"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        return self.time >= self.config["duration"]

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal (with or without penalties)."""
        rewards = self._agent_rewards(action, vehicle)
        reward_cpy = copy.deepcopy(rewards)
        if self.no_penalty_reward:
            del reward_cpy["collision_reward"]
            del reward_cpy["on_road_reward"]
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in reward_cpy.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward

        if self.no_penalty_reward:
            if self.config["normalize_reward"]:
                reward = utils.lmap(
                    reward,
                    [0, self.config["arrived_reward"]],
                    [0, 1],
                )
        else:
            # The default calculation: Penalty is included in the reward.
            reward *= rewards["on_road_reward"]
            if self.config["normalize_reward"]:
                reward = utils.lmap(
                    reward,
                    [self.config["collision_reward"], self.config["arrived_reward"]],
                    [0, 1],
                )
        return reward

    def seed(self, seed: int):
        self._np_random, seed = seeding.np_random(seed)


class FixedVehicleTruncateOnlyIntersectionEnv(TruncateOnlyIntersectionEnv):
    """An intersection env with a fixed type of vehicle on the road.

    Compared to the standard environment, this version does not randomize the vehicles
    behavior so all other vehicles have the same behavior parameters.

    Compared to TruncateOnlyIntersectionEnv, we only overwrite spawn_vehicle with a
    version that does not randomize the behavior (skipping a single line compared
    to IntersectionEnv).
    """

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))

        self.road.vehicles.append(vehicle)
        return vehicle


def register_highway_envs():
    """Register the self-defined highway environments."""
    register(
        id="Intersect-TruncateOnly-v0",
        entry_point="constraint_learning.envs.highway_envs:TruncateOnlyIntersectionEnv",
    )
    register(
        id="IntersectDefensive-TruncateOnly-v0",
        entry_point=(
            "constraint_learning.envs.highway_envs:"
            "FixedVehicleTruncateOnlyIntersectionEnv"
        ),
        kwargs={
            "config": {
                "other_vehicles_type": "highway_env.vehicle.behavior.DefensiveVehicle"
            }
        },
    )
    register(
        id="IntersectAggressive-TruncateOnly-v0",
        entry_point=(
            "constraint_learning.envs.highway_envs:"
            "FixedVehicleTruncateOnlyIntersectionEnv"
        ),
        kwargs={
            "config": {
                "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle"
            }
        },
    )
