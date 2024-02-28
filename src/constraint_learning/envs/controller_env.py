import dataclasses
import enum
import functools
from typing import Callable, Optional, TypeVar

import gymnasium
import numpy as np
from highway_env.envs.common import action
from highway_env.vehicle import behavior

Observation = TypeVar("Observation")


class IntersectionGoal(str, enum.Enum):
    LEFT = "o1"
    MIDDLE = "o2"
    RIGHT = "o3"

    @classmethod
    def random(cls):
        return np.random.choice(
            [IntersectionGoal.LEFT, IntersectionGoal.MIDDLE, IntersectionGoal.RIGHT]
        )

    def __str__(self):
        return self.value


@dataclasses.dataclass
class ControllerParameters:
    steering: np.ndarray
    acceleration: np.ndarray


class LinearVehicleIntersectionWrapper(gymnasium.core.Wrapper):
    """Intersection environment where agent is controlled by a linear controller.

    The agent is controlled by a linear controller with a set of acceleration and
    steering parameters, based on `highway_env.vehicle.behavior.LinearVehicle`.

    Inherits from `highway_envs.TruncateOnlyIntersectionEnv` to remove all termination
    conditions except truncation at config["duration"].

    To define a controller, you need to set the parameters via
        - `set_behavior_parameters()`
        - `randomize_behavior()`
    and set the goal via
        - `set_target_pos()`
        - `randomize_goal()`

    Reset the environment to spawn an agent vehicle with the parameters and goal.
    The environment ignores actions passed to `step()` and instead uses the controller.
    """

    # Defining the parameters as class attributes is not so nice, but it is consistent
    # with the way the LinearVehicle class is designed. We add the target position as
    # an additional parameter.
    ACCELERATION_PARAMETERS: np.ndarray = np.array(
        behavior.LinearVehicle.ACCELERATION_PARAMETERS
    )
    STEERING_PARAMETERS: np.ndarray = np.array(
        behavior.LinearVehicle.STEERING_PARAMETERS
    )
    ACCELERATION_RANGE: np.ndarray = behavior.LinearVehicle.ACCELERATION_RANGE
    STEERING_RANGE: np.ndarray = behavior.LinearVehicle.STEERING_RANGE
    GOAL: IntersectionGoal = IntersectionGoal.LEFT

    def __init__(
        self,
        env: gymnasium.Env,
        controller_paramters: Optional[ControllerParameters] = None,
        **kwargs
    ):
        super().__init__(env, **kwargs)
        if controller_paramters is not None:
            self.set_parameters(controller_paramters)

    def set_parameters(self, parameters: ControllerParameters) -> None:
        self.ACCELERATION_PARAMETERS = parameters.acceleration
        self.STEERING_PARAMETERS = parameters.steering

    def set_goal(self, goal: IntersectionGoal) -> None:
        self.GOAL = goal

    def get_random_parameters(self) -> ControllerParameters:
        """Gets random parameters from the ranges defines as class attributes."""
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        acceleration = self.ACCELERATION_RANGE[0] + ua * (
            self.ACCELERATION_RANGE[1] - self.ACCELERATION_RANGE[0]
        )
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        steering = self.STEERING_RANGE[0] + ub * (
            self.STEERING_RANGE[1] - self.STEERING_RANGE[0]
        )
        return ControllerParameters(steering=steering, acceleration=acceleration)

    def randomize_behavior(self) -> None:
        """Randomizes the controller parameters."""
        self.set_parameters(self.get_random_parameters())

    def randomize_goal(self) -> None:
        """Randomizes the goal for the controller."""
        self.set_goal(IntersectionGoal.random())

    def reset(self, **kwargs):
        return_value = super().reset(**kwargs)

        # Use a local class to define a LinearVehicle with the parameters we want
        class ParamVehicle(behavior.LinearVehicle):
            ACCELERATION_PARAMETERS = self.ACCELERATION_PARAMETERS
            STEERING_PARAMETERS = self.STEERING_PARAMETERS

        # The action type determinse which vehicle is spawned for the agent.
        # This dummy class makes sure we spawn our custom `ParamVehicle`.
        class DummyActionType(action.DiscreteMetaAction):
            @property
            def vehicle_class(self) -> Callable:
                return functools.partial(
                    ParamVehicle, target_speed=self.target_speeds[1]
                )

        env = self.env.unwrapped
        env.action_type = DummyActionType(self)
        env._make_road()
        env._make_vehicles()
        env.controlled_vehicles[0].plan_route_to(self.GOAL)

        return return_value
