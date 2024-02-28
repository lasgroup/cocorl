from typing import Optional

import gym
import highway_env
import highway_env.envs
import numpy as np


class IntersectionFeatureWrapper(gym.core.Wrapper):
    """Wraps the intersection env to add features for computing reward/constraint.

    - Writes a set of features into the info dictionary as info["features"]
    - Overwrites the environment reward with reward @ feature vector
    - Computes a constraint function as constraint @ feature vector minus thresholds

    Args:
        env: The environment to wrap.
        reward_parameters: A numpy array of reward parameters.
        constraint_parameters: An optional numpy array of constraint parameters.
        constraint_thresholds: An optional numpy array of constraint thresholds.
        goal_dist_threshold: The distance for goal indicators to activate.
        speed_limit: Vehicle speed limit.
        required_distance_to_front_vehicle: The minimum distance to the front vehicle.
    """

    FEATURE_NAMES = [
        "left_goal",
        "middle_goal",
        "right_goal",
        "vehicle_speed",
        "heading_angle",
        "speed_gt_limit",
        "too_close_to_front_vehicle",
        "collision",
        "not_on_street",
    ]

    def __init__(
        self,
        env: highway_env.envs.AbstractEnv,
        reward_parameters: np.ndarray,
        constraint_parameters: Optional[np.ndarray] = None,
        constraint_thresholds: Optional[np.ndarray] = None,
        goal_dist_threshold: float = 30,
        speed_limit: float = 15,
        required_distance_to_front_vehicle: float = 10,
        reduce_reward_by_cost: bool = False,
        modify_reward: bool = True,
    ):
        super().__init__(env)
        self.reward_parameters = reward_parameters
        self.constraint_parameters = constraint_parameters
        self.constraint_thresholds = constraint_thresholds
        self.goal_dist_threshold = goal_dist_threshold
        self.speed_limit = speed_limit
        self.required_distance_to_front_vehicle = required_distance_to_front_vehicle
        self.reduce_reward_by_cost = reduce_reward_by_cost
        self.modify_reward = modify_reward

    def step(self, action):
        """Performs a step and modifies the reward and info dictionary.

        Args:
            action: The action to take in the environment.

        Returns:
            The next state, modified reward, done flag, and modified info dictionary.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self.new_info(info)
        if self.modify_reward:
            reward = self.reward_parameters @ info["feature_vector"]
            if self.reduce_reward_by_cost:
                reward -= np.sum(info["cost"])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment and modifies the info dictionary."""
        obs, info = self.env.reset(**kwargs)
        info = self.new_info(info)
        return obs, info

    def new_info(self, info):
        features = self.compute_features()
        info["features"] = {
            name: value for name, value in zip(self.FEATURE_NAMES, features)
        }
        info["feature_vector"] = features

        if (
            self.constraint_parameters is not None
            and self.constraint_thresholds is not None
        ):
            constraint_function = (
                self.constraint_parameters @ features - self.constraint_thresholds
            )
            info["cost"] = constraint_function
        return info

    def compute_features(self) -> np.ndarray:
        """Computes features for the current state of the environment.

        Returns:
            np.ndarray: A numpy array of computed features.
        """
        assert isinstance(self.env.unwrapped, highway_env.envs.AbstractEnv)
        env = self.env.unwrapped
        assert env.road is not None

        vehicle_pos = env.vehicle.position
        vehicle_vel = env.vehicle.velocity
        vehicle_heading = env.vehicle.heading
        vehicle_speed = np.linalg.norm(vehicle_vel)
        lane = env.vehicle.lane

        left_goal_pos = env.road.network.get_lane(("o1", "ir1", 0)).start
        middle_goal_pos = env.road.network.get_lane(("o2", "ir2", 0)).start
        right_goal_pos = env.road.network.get_lane(("o3", "ir3", 0)).start

        left_goal_dist = np.linalg.norm(vehicle_pos - left_goal_pos)
        middle_goal_dist = np.linalg.norm(vehicle_pos - middle_goal_pos)
        right_goal_dist = np.linalg.norm(vehicle_pos - right_goal_pos)
        left_goal = left_goal_dist < self.goal_dist_threshold
        middle_goal = middle_goal_dist < self.goal_dist_threshold
        right_goal = right_goal_dist < self.goal_dist_threshold

        speed_gt_limit = vehicle_speed > self.speed_limit

        front_vehicle, _ = env.road.neighbour_vehicles(
            env.vehicle, env.vehicle.lane_index
        )
        dist_to_front_vehicle = env.vehicle.lane_distance_to(front_vehicle)
        too_close_to_front_vehicle = (
            dist_to_front_vehicle < self.required_distance_to_front_vehicle
        )

        collision = env.vehicle.crashed
        not_on_street = not lane.on_lane(vehicle_pos)

        long, _ = lane.local_coordinates(vehicle_pos)
        lane_heading = lane.heading_at(long)
        heading_angle = np.mod(np.abs(vehicle_heading - lane_heading), 2 * np.pi)

        return np.array(
            [
                left_goal,
                middle_goal,
                right_goal,
                vehicle_speed,
                heading_angle,
                speed_gt_limit,
                too_close_to_front_vehicle,
                collision,
                not_on_street,
            ]
        )
