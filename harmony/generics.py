from __future__ import annotations

import typing
from dataclasses import dataclass

import pandas as pd

T = typing.TypeVar('T')


@dataclass
class Sequential:
    _seq: typing.Sequence[T]

    @classmethod
    def from_sequence(cls, sequence: typing.Sequence[T]) -> typing.Self:
        first_object_type = type(sequence[0])
        type_check_pass = all((type(x) == first_object_type for x in sequence))
        if not type_check_pass:
            raise TypeError()
        return cls(_seq=sequence)

    def get_n_grams(self, n: int) -> Sequential:
        length = len(self._seq)
        n_grams = [tuple(self._seq[i:i + n]) for i in range(length - n + 1)]
        n_grams = Sequential.from_sequence(sequence=n_grams)
        return n_grams

    def get_transition_matrix(self) -> pd.DataFrame:
        unique_objects = list(set(self._seq))
        unique_objects = list(map(str, unique_objects))
        transition_matrix = pd.DataFrame(0, columns=unique_objects, index=unique_objects)

        bigrams = self.get_n_grams(n=2)
        for bigram in bigrams._seq:
            source, target = bigram
            transition_matrix.loc[str(source), str(target)] += 1
        return transition_matrix


def test_sequential_get_n_grams():
    n: int = 2
    sequence = ['1', 'sdf', '333', '8', '9']
    # sequence = [1, [4], '333', lambda x:x+1, set()]

    sequential = Sequential.from_sequence(sequence=sequence)
    n_gram = sequential.get_n_grams(n=n)
    expected_length = len(sequence) - n + 1
    print(n_gram)
    assert len(n_gram) == expected_length


def test_sequential_get_transition_matrix():
    n: int = 2
    sequence = [1, 2, 3, 4, 1, 2, 3, 4, 6, 5, 6, 5, 6, 5, 4, 3, 2, 1]
    sequence = list(map(str, sequence))
    # sequence = [1, [4], '333', lambda x:x+1, set()]

    sequential = Sequential.from_sequence(sequence=sequence)
    matrix = sequential.get_n_grams(3).get_transition_matrix()
    print(matrix.to_markdown())


if __name__ == '__main__':
    test_sequential_get_transition_matrix()
