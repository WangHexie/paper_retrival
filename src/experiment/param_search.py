import argparse
import os
from typing import List

import elasticsearch
import yaml
import numpy as np
import copy
from ..config import root_path
from ..application.retrieve_for_evaluation import Retrieve, EmbeddingRetrieve
from time import sleep
"""
该参数搜索先将字典展平后计算list个数，只计算作为字典value的list而不深入计算list中的list。
统计完list个数和长度后，将长度相乘得到需要搜索的参数组合数量。
将list长度按出现顺序排列，将其作为进位标识符，表示每个位都具有不同的base，进位不同。
搜索所有参数只要把参数组合数量转换为自定义进制的数即可。该转换过程从零开始依次加一至所需的的参数数量值，依次进位，每个位即是改取的参数的位置。所以能保证搜索到所有参数。
将展平后的list与原始字典对应则直接使用self-called（刚编的名字） 函数，依次将参数传入对应。
"""

def read_yaml(name):
    with open(os.path.join(root_path, "config", name), "r") as f:
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
    """
    read yaml and
    :param config_name:
    :return:
    """
    config = read_yaml(config_name)
    custom_base = flatten_config_and_calculate_base(config)
    num_of_runs = calculate_number_of_runs(config)
    for i in range(num_of_runs):
        number_of_custom_base = output_number_for_the_given_base(i, custom_base)
        yield create_param_from_custom_base_numbers(number_of_custom_base, copy.deepcopy(config))


def save_results(text, config_name):
    with open(os.path.join(root_path, "logs", config_name + ".txt"), "a") as f:
        f.write(str(text))
        f.write("\n|||\n")


def retrieve_param_search(config_name, mode=0):
    for param in create_param_from_config(config_name):
        print(param)
        save_results(param, config_name)

        if mode == 0:
            rt = Retrieve("", "", **param)
        elif mode == 1:
            rt = EmbeddingRetrieve(**param)
        try:
            result = rt.evaluate()
        except elasticsearch.exceptions.TransportError as e:
            print(e.info)
        rt.close()

        print(result)
        save_results(result, config_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='param search')
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', type=int, default=0)

    args = parser.parse_args()
    retrieve_param_search(args.config, mode=args.mode)

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
