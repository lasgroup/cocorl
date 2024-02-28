import glob
import json
import random


def load_demos(demonstration_folder: str, num_demos: int):
    demonstration_files = glob.glob(f"{demonstration_folder}/*.json")
    files_sample = random.sample(demonstration_files, num_demos)

    data_list = []
    for filename in files_sample:
        with open(filename, "r") as f:
            data = json.load(f)
            data_list.append(data)
    return data_list
