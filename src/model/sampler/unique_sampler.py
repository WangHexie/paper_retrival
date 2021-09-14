import itertools
import random
from itertools import zip_longest
from random import shuffle
from typing import List, Iterator

from torch.utils.data import Sampler


class NoduplicateSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: List, batch_size) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.full_list = self._create_full_list()
        # self._count_num_of_class()

    def _count_num_of_class(self):
        classes = list(map(lambda x: x.label, self.data_source))
        num_of_class = max(classes) - min(classes) + 1
        assert min(classes) == 0
        samples = [[] for i in range(num_of_class)]
        for i in range(len(self.data_source)):
            samples[self.data_source[i].label].append(i)

        for i in range(len(samples)):
            shuffle(samples[i])

        shuffle(samples)

        self.samples = samples

    def _create_full_list(self):
        self._count_num_of_class()

        r = [list(filter(None, i)) for i in zip_longest(*self.samples)]
        r = list(filter(lambda x: len(x) >= self.batch_size, r))
        full_list = list(itertools.chain.from_iterable(r))
        return full_list

    def __iter__(self) -> Iterator[int]:
        self.full_list = self._create_full_list()
        return iter(self.full_list)

    def __len__(self) -> int:
        return len(self.full_list)


class TestSampler:

    def __init__(self) -> None:
        pass

    def __iter__(self):
        r = [random.randint(1, 10) for i in range(10)]

        return iter(r)
