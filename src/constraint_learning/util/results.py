import dataclasses
import os
from datetime import datetime

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np

jsonpickle_numpy.register_handlers()


@dataclasses.dataclass
class SyntheticExperimentResult:
    """Contains the result of an experiment with synthetic data.

    Attributes:
        - true_reward: reward of optimal solution with ground truth constraint
        - safe_reward: reward achieved by constraint learning algorithm
        - true_solution_in_safe_set: fraction of true solutions in inferred safe set
        - true_solution_in_unsafe_set: fraction of true solutions in inferred unsafe set
        - true_solution_in_uncertain_set: fraction of true solutions neither in inferred
                                          safe or unsafe set
        - safe_set_size: size of the safe set
        - unsafe_set_size: size of the unsafe set
        - uncertain_set_size: size of uncertain set
    """

    true_reward: np.ndarray
    safe_reward: np.ndarray
    min_reward: np.ndarray
    found_safe_solution: np.ndarray
    safe_constraint_violations: np.ndarray
    safe_constraint_distance: np.ndarray
    safe_inferred_constraint_violations: np.ndarray
    safe_inferred_constraint_distance: np.ndarray
    true_solution_in_safe_set: np.ndarray
    true_solution_in_unsafe_set: np.ndarray
    true_solution_in_uncertain_set: np.ndarray
    safe_set_size: float
    unsafe_set_size: float
    uncertain_set_size: float


@dataclasses.dataclass
class HighwayExperimentResult:
    """Contains result of a highway experiment with discrete policies.

    Attributes:
        - true_reward: reward of optimal solution with ground truth constraint
        - safe_reward: reward achieved by constraint learning algorithm
        - irl_reward: reward achieved by (maximum margin) IRL baseline
        - safe_constraint_violations: constraint violations of solution
        - irl_constraint_violations: constraint violations of IRL baseline
        - true_solution_in_safe_set: fraction of true solutions in inferred safe set
        - true_solution_in_unsafe_set: fraction of true solutions in inferred unsafe set
        - true_solution_in_uncertain_set: fraction of true solutions neither in inferred
                                          safe or unsafe set
        - safe_set_size: size of the safe set
        - unsafe_set_size: size of the unsafe set
        - uncertain_set_size: size of uncertain set
    """

    true_reward: np.ndarray
    safe_reward: np.ndarray
    irl_reward: np.ndarray
    safe_constraint_violations: np.ndarray
    irl_constraint_violations: np.ndarray
    true_solution_in_safe_set: np.ndarray
    true_solution_in_unsafe_set: np.ndarray
    true_solution_in_uncertain_set: np.ndarray
    safe_set_size: float
    unsafe_set_size: float
    uncertain_set_size: float


@dataclasses.dataclass
class CEHighwayExperimentResult:
    """Contains the result of a highway experiment using the cross-entropy solver.

    Attributes:
        - true_reward: reward of optimal solution with ground truth constraint
        - safe_reward: reward achieved by constraint learning algorithm
        - found_safe_solution: boolean whether we found a safe solution for new rewards
        - safe_constraint_violations: constraint violations of solution
        - true_solution_in_safe_set: fraction of true solutions in inferred safe set
        - true_solution_in_unsafe_set: fraction of true solutions in inferred unsafe set
        - true_solution_in_uncertain_set: fraction of true solutions neither in inferred
                                          safe or unsafe set
    """

    true_reward: np.ndarray
    safe_reward: np.ndarray
    found_safe_solution: np.ndarray
    safe_constraint_violations: np.ndarray
    true_solution_in_safe_set: np.ndarray
    true_solution_in_unsafe_set: np.ndarray
    true_solution_in_uncertain_set: np.ndarray


@dataclasses.dataclass
class GridworldExperimentResult:
    """Contains the result of a Gridworld experiment using a linear programming solver.

    Attributes:
        - true_reward: reward of optimal solution with ground truth constraint
        - safe_reward: reward achieved by constraint learning algorithm
        - safe_constraint_violations: constraint violations of solution
        - safe_inferred_constraint_violations: inferred constraint violations of
            solution (should be 0 if LP solver found a feasible solution)
        - safe_constraint_max: maximum constraint violation of inferred constraints
        - true_solution_in_safe_set: fraction of true solutions in inferred safe set
        - true_solution_in_unsafe_set: fraction of true solutions in inferred unsafe set
        - true_solution_in_uncertain_set: fraction of true solutions neither in inferred
                                          safe or unsafe set
        - found_safe_solution: boolean whether we found a safe solution for new rewards
        - num_inferred_constraints: number of inferred constraints
    """

    true_reward: np.ndarray
    safe_reward: np.ndarray
    min_reward: np.ndarray
    safe_constraint_violations: np.ndarray
    safe_constraint_distance: np.ndarray
    safe_constraint_max: np.ndarray
    safe_inferred_constraint_violations: np.ndarray
    safe_inferred_constraint_distance: np.ndarray
    true_solution_in_safe_set: np.ndarray
    true_solution_in_unsafe_set: np.ndarray
    true_solution_in_uncertain_set: np.ndarray
    found_safe_solution: np.ndarray
    num_inferred_constraints: np.ndarray


class Artifact:
    def __init__(self, file_name, method, _run):
        self._run = _run
        self.file_name = file_name
        self.file_path = os.path.join("/tmp", file_name)
        if method is not None:
            self.file_obj = open(self.file_path, method)
        else:
            self.file_obj = None

    def __enter__(self):
        if self.file_obj is None:
            return self.file_path
        else:
            return self.file_obj

    def __exit__(self, type, value, traceback):
        if self.file_obj is not None:
            self.file_obj.close()
        self._run.add_artifact(self.file_path)


def _read_json(result_folder, filename):
    try:
        with open(os.path.join(result_folder, filename), "r") as f:
            json_str = f.read()
        return jsonpickle.loads(json_str)
    except Exception as e:
        print(e)
        return None


class FileExperimentResults:
    def __init__(self, result_folder):
        self.result_folder = result_folder
        self.config = _read_json(result_folder, "config.json")
        self.metrics = _read_json(result_folder, "metrics.json")
        self.run = _read_json(result_folder, "run.json")
        self.info = _read_json(result_folder, "info.json")
        assert isinstance(self.run, dict)
        self.status = self.run["status"]
        self.result = self.run["result"]

    def _convert_str_to_time(self, time_str):
        try:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")

    def _get_start_time(self):
        assert isinstance(self.run, dict)
        return self._convert_str_to_time(self.run["start_time"])

    def _get_stop_time(self):
        assert isinstance(self.run, dict)
        return self._convert_str_to_time(self.run["stop_time"])

    def get_metric(self, name, get_runtime=False):
        assert isinstance(self.metrics, dict)
        metric = self.metrics[name]
        if get_runtime:
            start_time = self._get_start_time()
            timestamps = [self._convert_str_to_time(t) for t in metric["timestamps"]]
            run_time = [(t - start_time).total_seconds() for t in timestamps]
            return metric["steps"], metric["values"], run_time
        else:
            return metric["steps"], metric["values"]

    def print_captured_output(self):
        with open(os.path.join(self.result_folder, "cout.txt"), "r") as f:
            print(f.read())

    def get_runtime(self):
        start_time = self._get_start_time()
        stop_time = self._get_stop_time()
        return (stop_time - start_time).total_seconds()
