from __future__ import annotations
import typing
from typing import Callable, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from musana.harmony_types import TonalHarmony

metrics = {}
A = typing.TypeVar('A')
B = typing.TypeVar('B')
C = typing.TypeVar('C')


def maybe_bind(f: Callable[[A], Optional[B]], g: Callable[[B], Optional[C]]) -> Callable[[A], Optional[C]]:
    def h(x: A) -> Optional[C]:
        y = f(x)
        if y is None:
            #print('y is None')
            return None
        else:
            #print(f'{y.chord_str} is valid')
            return g(y)

    return h


@dataclass
class ChromaticityMetrics:

    @staticmethod
    def metric1(chord: TonalHarmony) -> typing.Dict[str, int]:
        assert isinstance(chord, TonalHarmony)

        non_diatonic_sds_in_globalkey = chord.non_diatonic_sds(reference_key=chord.globalkey)
        d_to_global_tonic = sum([abs(sd.to_spc().fifths()) for sd in non_diatonic_sds_in_globalkey])

        non_diatonic_sds_in_localkey = chord.non_diatonic_sds(reference_key=chord.localkey)
        d_to_local_tonic = sum([abs(sd.to_spc().fifths()) for sd in non_diatonic_sds_in_localkey])

        global_tonic = chord.globalkey.to_str()
        local_tonic = chord.localkey.to_str()

        result_dict = {'chord': chord.chord_str,
                       'pcs': chord.pcs,
                       'global_tonic': global_tonic,
                       'd_to_global': d_to_global_tonic,
                       'local_tonic': local_tonic,
                       'd_to_local': d_to_local_tonic}

        return result_dict

    @staticmethod
    def metric1_df(piece: pd.DataFrame) -> pd.DataFrame:
        row_func = lambda row: pd.Series(maybe_bind(TonalHarmony.from_df_row, ChromaticityMetrics.metric1)(row), dtype='object')

        result1: pd.DataFrame = piece.apply(row_func, axis=1)
        # nan_rows = result1.isnull().any(axis=1)
        # nan_indices = result1.index[nan_rows]
        result = result1.dropna()
        # unpacked_df = pd.DataFrame(result.apply(lambda row: pd.Series(row)))
        return result


def test_func(row: pd.DataFrame):
    return (row['globalkey'], row['localkey'])


def to_run():
    df = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/tchaikovsky_seasons/harmonies/op37a12.tsv',
        sep='\t')

    result = ChromaticityMetrics.metric1_df(piece=df)
    print(f'{result=}')


if __name__ == '__main__':
    to_run()
