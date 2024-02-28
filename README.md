# Constraint learning from demonstrations with unknown rewards

This repository contains code for the paper ["Learning Safety Constraints from Demonstrations with Unknown Rewards"](https://arxiv.org/abs/2305.16147). Here, we describe how to reproduce the experiments presented in the paper.


### Citation

David Lindner, Xin Chen, Sebastian Tschiatschek, Katja Hofmann, Andreas Krause. **Learning Safety Constraints from Demonstrations with Unknown Rewards**. In _International Conference on Artificial Intelligence and Statistics (AISTATS)_, 2024.

```
@inproceedings{lindner2024learning,
    title={Learning Safety Constraints from Demonstrations with Unknown Rewards},
    author={Lindner, David and Chen, Xin and Tschiatschek, Sebastian and Hofmann, Katja and Krause, Andreas},
    booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
    year={2024},
}
```


## Setup

We recommend using Anaconda to set up an environment with the dependencies of this repository. Run the following commands from this repository to set up the environment:

```
conda create -n cocorl python=3.9
conda activate cocorl
pip install -e .
```

This sets up a Anaconda environment with the required dependencies and activates it.


## Single-state MDP Experiments (Appendix G.1)

To run the experiments with synthetic single-state CMDP instances, use
```
python src/constraint_learning/linear/synthetic_experiment.py
```
Parameters and experiment tracking are handled via `sacred`. Results are stored in `results/`, and arguments can by passed using `with`. A sweep of experiments is defined in `experiment_configs/synthetic/synthetic.json`, and it can be run with
```
python run_sacred_experiments.py --config experiment_configs/synthetic/synthetic.json --num_jobs 1
```

The `--num_jobs` flag can be used to parallelize the runs over multiple CPUs.


## Gridworld Experiments (Section 6.2)

To run the Gridworld experiments, use
```
python src/constraint_learning/linear/gridworld_experiment.py
```

In the paper, we present 3 different experiments to test reward transfer. These can be run with the following three parameter sweeps:
```
python run_sacred_experiments.py --config experiment_configs/gridworld/gridworld_exp1_no_transfer.json --num_jobs 1
python run_sacred_experiments.py --config experiment_configs/gridworld/gridworld_exp2_reward_transfer.json --num_jobs 1
python run_sacred_experiments.py --config experiment_configs/gridworld/gridworld_exp3_env_transfer.json --num_jobs 1
```

Note that these sweeps run a lot of random seeds by default. You'll likely want to increase the number of parallel jobs using `--num_jobs`.


## HighwayEnv Experiments (Section 6.3)

To run experiments in `highway-env`, we first need to precompute potential demonstrations for different reward functions (in the standard environment, as well as environments with aggressive and defensive drivers). To do that, run
```
python run_sacred_experiments.py --config experiment_configs/highway_ce/ce_demonstrations.json --num_jobs 1
python run_sacred_experiments.py --config experiment_configs/highway_ce/ce_demonstrations_aggressive.json --num_jobs 1
python run_sacred_experiments.py --config experiment_configs/highway_ce/ce_demonstrations_defensive.json --num_jobs 1
```
For convenience, we provide a set of precomputed demonstrations. To use them, unpack the archive `demonstrations.zip` into the folder `demonstrations/`.

Now, experiments can be run using the experiment script:
```
python src/constraint_learning/linear/highway_experiment.py
```

To run the three experiments presented in the paper, execute the following commands:
```
python run_sacred_experiments.py --config experiment_configs/highway_ce/exp1_no_transfer.json --num_jobs 1
python run_sacred_experiments.py --config experiment_configs/highway_ce/exp2_goal_transfer.json --num_jobs 1
python run_sacred_experiments.py --config experiment_configs/highway_ce/exp3_env_transfer.json --num_jobs 1
```


## Aggregating experiment results

By default the results of all experiments are stored in `results/`. Each result is stored in an individual sub-folder labelled with a timestamp. To aggregate and evaluate the experiment results, we provide a script `aggregate_results.py`. To use it, execute
```
python scripts/aggregate_results.py --results_folder results/ --out_file aggregated.csv
```

This produces a csv file `aggregated.csv` containing the results from all experiments in `results/`. This csv file can be used to reproduce the plots shown in the paper.


## Unit Tests

We use `pytest` for unit tests. Run all tests with `pytest .`


## Linting

We use `black`, `isort`, `flake8`, `mypy`, `darglint`, and `jsonlint-php`. Run all linters with `lint.sh`.
