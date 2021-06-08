import argparse
import os
from typing import List

import yaml
import numpy as np


def read_yaml(name):
    with open(os.path.join("config", name), "r") as f:
        config = yaml.load(f)
    return config


def calculate_number_of_runs(config) -> int:
    if type(config) == list:
        return len(config)
    else:
        nums = [calculate_number_of_runs(value) for key, value in config.items()]
        return np.prod(nums)


def flatten_config_and_calculate_base(config) -> List[int]:
    if type(config) == list:
        return [len(config)]
    else:
        nums = [calculate_number_of_runs(value) for key, value in config.items()]
        results = []
        for i in nums:
            results += i
        return results


def output_number_for_the_given_base(num, base: List[int]):
    def calculate_carry(number_of_base, given_base):
        assert len(number_of_base) == len(given_base)
        for i in range(len(number_of_base)):
            if number_of_base[i] == given_base[i]:
                number_of_base[i] -= 1
                number_of_base[i + 1] += 1
        return number_of_base

    result = [0 for i in range(len(base))]
    for i in range(num):
        result[0] +=1
        result = calculate_carry(result, base)

    return result

# def create_param

def create_param_from_config(config, number):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='param search')
    parser.add_argument('--config', type=str)

    args = parser.parse_args()
    print(read_yaml(args.config))
    print(calculate_number_of_runs(read_yaml(args.config)))
    print(flatten_config_and_calculate_base(read_yaml(args.config)))

# def
