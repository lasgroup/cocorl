import argparse
import datetime
import os
import subprocess

import pandas as pd

from constraint_learning.util import results


def key_or_none(d, key):
    if d is not None and key in d:
        return d[key]
    return None


def load(results_folder, experiment_label, config_query=None):
    if experiment_label is not None:
        result = subprocess.check_output(
            f"grep '\"{experiment_label}\"' -r {results_folder} "
            "| grep config | cut -f 1 -d : | rev | cut -d / -f 2- | rev",
            shell=True,
        ).decode()
        subdirs = result.split("\n")
    else:
        subdirs = [x[0] for x in os.walk(results_folder)]

    for i, subdir in enumerate(subdirs):
        print(i, subdir)
        try:
            experiment = results.FileExperimentResults(subdir)
        except Exception as e:
            print(e)
            continue
        valid = True
        if config_query is not None:
            for key, value in config_query.items():
                if experiment.config[key] != value:
                    valid = False
        if valid:
            yield experiment


def get_df_entry_from_experiment(ex):
    if ex.status == "COMPLETED":
        try:
            num_new_thetas = len(ex.result.true_reward)
            for i in range(num_new_thetas):
                new_theta_range = key_or_none(ex.info, "new_theta_ranges")
                method = key_or_none(ex.config, "method")

                result = dict(
                    method=method,
                    num_dims=key_or_none(ex.config, "num_dims"),
                    num_thetas=key_or_none(ex.config, "num_thetas"),
                    num_new_thetas=key_or_none(ex.config, "num_new_thetas"),
                    num_phis=key_or_none(ex.config, "num_phis"),
                    seed=key_or_none(ex.config, "seed"),
                    true_reward=ex.result.true_reward[i],
                    safe_reward=ex.result.safe_reward[i],
                    safe_constraint_violation=ex.result.safe_constraint_violations[i],
                    true_solution_in_safe_set=(ex.result.true_solution_in_safe_set[i]),
                    true_solution_in_unsafe_set=(
                        ex.result.true_solution_in_unsafe_set[i]
                    ),
                    true_solution_in_uncertain_set=(
                        ex.result.true_solution_in_uncertain_set[i]
                    ),
                    new_thetas_range=None
                    if new_theta_range is None
                    else new_theta_range[i],
                    result_folder=ex.result_folder,
                    new_theta_seed=key_or_none(ex.config, "new_theta_seed"),
                    model_weight_file=key_or_none(ex.config, "model_weight_file"),
                )

                synthetic = isinstance(ex.result, results.SyntheticExperimentResult)
                highway = isinstance(ex.result, results.HighwayExperimentResult)
                highway_ce = isinstance(ex.result, results.CEHighwayExperimentResult)
                gridworld = isinstance(ex.result, results.GridworldExperimentResult)

                if synthetic:
                    result["active_learning"] = ex.config["active_learning"]

                if synthetic or highway:
                    result["safe_set_size"] = ex.result.safe_set_size
                    result["unsafe_set_size"] = ex.result.unsafe_set_size
                    result["uncertain_set_size"] = ex.result.uncertain_set_size

                if highway:
                    result["irl_reward"] = ex.result.irl_reward[i]
                    result[
                        "irl_constraint_violation"
                    ] = ex.result.irl_constraint_violations[i]

                if (
                    synthetic
                    or highway_ce
                    or (gridworld and method == "constraint_learning")
                ):
                    result["found_safe_solution"] = ex.result.found_safe_solution[i]

                if gridworld:
                    result["safe_constraint_max"] = ex.result.safe_constraint_max[i]

                if gridworld and method == "constraint_learning":
                    result[
                        "num_inferred_constraints"
                    ] = ex.result.num_inferred_constraints

                if synthetic or gridworld:
                    result["min_reward"] = ex.result.min_reward[i]
                    result[
                        "safe_constraint_distance"
                    ] = ex.result.safe_constraint_distance[i]
                    if synthetic or method == "constraint_learning":
                        result[
                            "safe_inferred_constraint_distance"
                        ] = ex.result.safe_inferred_constraint_distance[i]
                        result[
                            "safe_inferred_constraint_violations"
                        ] = ex.result.safe_inferred_constraint_violations[i]

                return result
        except Exception as e:
            print("Exception:", e)
    else:
        print("WARNING: Experiment not completed. Status:", ex.status)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, default="results/")
    parser.add_argument("--remote_results_folder", type=str, default=None)
    parser.add_argument("--tmp_folder", type=str, default="/tmp")
    parser.add_argument("--experiment_label", type=str, default=None)
    parser.add_argument("--out_file", type=str, default="results.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.results_folder is None and args.remote_results_folder is None:
        raise ValueError("results_folder or remote_results_folder has to be given")

    if args.remote_results_folder is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tmp_result_folder = os.path.join(args.tmp_folder, timestamp)
        subprocess.run(
            [
                "rsync",
                "-av",
                "-e ssh",
                "--exclude",
                "cout.txt",
                "--exclude",
                "metrics.json",
                "--exclude",
                "*.pkl",
                args.remote_results_folder,
                tmp_result_folder,
            ]
        )
        args.results_folder = tmp_result_folder

    experiments = load(args.results_folder, args.experiment_label)

    results_df = pd.DataFrame(
        (
            entry
            for entry in (get_df_entry_from_experiment(ex) for ex in experiments)
            if entry is not None
        )
    )
    results_df.to_csv(args.out_file, index=False)

    print(f"Saved aggregated results to {args.out_file}")


if __name__ == "__main__":
    main()
