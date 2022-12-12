# Created by Xinyi Guan in 2022.
from __future__ import annotations

import collections
import typing
from dataclasses import dataclass

import pandas as pd

T = typing.TypeVar('T')


@dataclass
class Sequential:
    _seq: typing.Sequence[T]

    @classmethod
    def from_sequence(cls, sequence: typing.Sequence[T]) -> typing.Self:
        if len(sequence) > 0:
            first_object_type = type(sequence[0])
            type_check_pass = all((type(x) == first_object_type for x in sequence))
            if not type_check_pass:
                raise TypeError()
        return cls(_seq=sequence)

    @classmethod
    def join(cls, sequentials: typing.Sequence[Sequential]) -> Sequential:
        """
        A method to join multiple sequentials.
        :param sequentials:
        :return:
        """
        joined_seq = sum([x._seq for x in sequentials], [])
        sequential = cls.from_sequence(joined_seq)
        return sequential

    def map(self, operation: typing.Callable[[T, ], typing.Any]) -> Sequential:
        new_seq = list(map(operation, self._seq))
        sequential = Sequential(_seq=new_seq)
        return sequential

    def get_n_grams(self, n: int) -> Sequential:
        """
        Returns a Squential type of object in which each transition is expressed as a tuple.
        :param n:
        :return:
        """
        length = len(self._seq)
        n_grams = [tuple(self._seq[i:i + n]) for i in range(length - n + 1)]
        n_grams = Sequential.from_sequence(sequence=n_grams)
        return n_grams

    def get_transition_matrix(self, probability: bool) -> pd.DataFrame:
        """
        Create a transition matrix
        :param n:
        :param probability:
        :return:
        """

        count = collections.Counter(self._seq)
        top_common_objects = [x for x, number in count.most_common()]

        transition_matrix = pd.DataFrame(0, columns=top_common_objects, index=top_common_objects)

        bigrams = self.get_n_grams(n=2)
        for bigram in bigrams._seq:
            source, target = bigram
            transition_matrix.loc[str(source), str(target)] += 1

        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix

    def filter_by_condition(self, condition: typing.Callable[[T, ], bool]) -> Sequential:
        sequence = [x for x in self._seq if condition(x)]
        sequential = self.from_sequence(sequence=sequence)
        return sequential


if __name__ == '__main__':
    pass
    # test_seq = [1, 1, 1, 2, 2, 3, 4, 5, 6]
    # sequential = Sequential.from_sequence(sequence=test_seq)
    # bigrams = sequential.get_n_grams(n=2)
    # print(bigrams)
