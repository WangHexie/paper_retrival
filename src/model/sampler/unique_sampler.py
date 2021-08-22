import itertools
from itertools import zip_longest
from random import shuffle
from typing import List, Iterator

from torch.utils.data import Sampler


class NoduplicateSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: List) -> None:
        self.data_source = data_source
        self._count_num_of_class()

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

    def __iter__(self) -> Iterator[int]:

        r = [list(filter(None, i)) for i in zip_longest(*self.samples)]

        return iter(list(itertools.chain.from_iterable(r)))

    def __len__(self) -> int:
        return len(self.data_source)
