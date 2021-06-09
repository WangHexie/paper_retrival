import argparse
import os
from typing import List

import yaml
import numpy as np
import copy

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
        return [len(config), ]
    else:
        nums = [flatten_config_and_calculate_base(value) for key, value in config.items()]
        results = []
        for i in nums:
            results += i
        return results


def output_number_for_the_given_base(num, base: List[int]):
    def calculate_carry(number_of_base, given_base):
        assert len(number_of_base) == len(given_base)
        for i in range(len(number_of_base)):
            if number_of_base[i] == given_base[i]:
                number_of_base[i] = 0
                number_of_base[i + 1] += 1
        return number_of_base

    result = [0 for i in range(len(base))]
    for i in range(num):
        result[0] += 1
        result = calculate_carry(result, base)

    return result


def create_param_from_custom_base_numbers(number: List[int], config: dict or list):
    if type(config) == list:
        index = number.pop(0)
        return config[index]
    else:
        for key, value in config.items():
            config[key] = create_param_from_custom_base_numbers(number, config[key])
        return config


def create_param_from_config(config_name):
    config = read_yaml(config_name)
    custom_base = flatten_config_and_calculate_base(config)
    num_of_runs = calculate_number_of_runs(config)
    for i in range(num_of_runs):
        number_of_custom_base = output_number_for_the_given_base(i, custom_base)
        yield create_param_from_custom_base_numbers(number_of_custom_base, copy.deepcopy(config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='param search')
    parser.add_argument('--config', type=str)

    args = parser.parse_args()
    for i in create_param_from_config(args.config):
        print(i)
    # print(read_yaml(args.config))
    # print(calculate_number_of_runs(read_yaml(args.config)))
    # print(flatten_config_and_calculate_base(read_yaml(args.config)))
    # for i in range(16):
    #     print(output_number_for_the_given_base(i, flatten_config_and_calculate_base(read_yaml(args.config))))
    #     config = read_yaml(args.config)
    #     custom_base = flatten_config_and_calculate_base(config)
    #     number_of_custom_base = output_number_for_the_given_base(i, custom_base)
    #     print(create_param_from_custom_base_numbers(number_of_custom_base, config))

# def
