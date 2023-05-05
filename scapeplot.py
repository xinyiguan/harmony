from dataclasses import dataclass
from math import isclose
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from pitchtypes import SpelledPitchClass

from harmonytypes.numeral import Numeral
from harmonytypes.util import maybe_bind
from main import piece_wise_operation, piecewise_pc_content_indices_df


@dataclass
class Scape:
    """Abstract base class for scapes objects. Which has a minimum time (min_time) a maximum time (max_time) and a
    well-defined value for any window (t1, t2) with min_time <= t1 < t2 <= max_time. The value can be retrieved via
    scape[t1, t2]."""

    min_time: int
    max_time: int

    def __getitem__(self, item: Tuple[int, int]):
        """
        Return value of the scape at position
        :param item: (start, end) with start < end
        :return: value of the scape for the time window (start, end)
        """
        raise NotImplementedError

    def assert_valid_time_window(self, start, end):
        if start < self.min_time and not isclose(start, self.min_time):
            raise ValueError(f"Window start is less than minimum time ({start} < {self.min_time})")
        if end > self.max_time and not isclose(end, self.max_time):
            raise ValueError(f"Window end is greater than maximum time ({end} > {self.max_time})")
        if start > end:
            raise ValueError(f"Window start is greater than or equal to window end ({start} >= {end})")


def all_window_size_grams_from_ndarray(array: np.ndarray) -> List[Tuple]:
    n_grams = []
    for n in range(1, len(array) + 1):
        for i in range(len(array) - n + 1):
            n_grams.append(tuple(array[i:i + n]))
    return n_grams


def piecewise_all_window_size_grams(piece_df: pd.DataFrame, chord_wise_operation: Optional = lambda x: x) -> List[
    Tuple]:
    numerals_array = piece_wise_operation(piece_df=piece_df, chord_wise_operation=chord_wise_operation).to_numpy()
    result = all_window_size_grams_from_ndarray(array=numerals_array)
    return result


def piecewise_unigram_pc_content_index(piece_df: pd.DataFrame):
    result = piecewise_pc_content_indices_df(piece_df=piece_df)
    return result


def recursive_adjacent_sum(l: List[int]) -> List[int]:
    if len(l) == 1:
        return l
    return l + recursive_adjacent_sum([l[i] + l[i + 1] for i in range(len(l) - 1)])


if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')

    result = piecewise_all_window_size_grams(piece_df=df, chord_wise_operation=ndt_in_chord)
    print(f'{len(result)}')
