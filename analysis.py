from __future__ import annotations
import collections
import typing
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import musana.generics as generics
from musana.loader import PieceInfo, MetaCorporaInfo, CorpusInfo
from musana.harmony_types import TonalHarmony
import musana.plotting as plotting
from musana.metrics import NonDiatonicPCs

metrics = {}


@dataclass
class ChromaticityMetrics:


    @staticmethod
    def chord_chromaticity(chord: TonalHarmony) -> typing.Dict[str, int]:
        """refernece tonics (3 levels): global tonic, local tonic and pseduo tonic"""
        pcs = chord.pcs
        d_to_global_tonic = NonDiatonicPCs(pcs=pcs, reference_key=chord.globalkey).sum_in_fifths()
        d_to_local_tonic = NonDiatonicPCs(pcs=pcs, reference_key=chord.globalkey).sum_in_fifths()
        result_dict = {'d_to_global_tonic': d_to_global_tonic,
                       'd_to_local_tonic': d_to_local_tonic}
        return result_dict

    @staticmethod
    def df_chord_chromaticity(piece: pd.DataFrame) -> pd.DataFrame:
        row_func = lambda row: ChromaticityMetrics.chord_chromaticity(TonalHarmony.from_df_row(row))
        result = piece.apply(row_func,axis=1)

        return result

def test_func(row:pd.DataFrame):
    return (row['globalkey'],row['localkey'])



if __name__ == '__main__':
    df = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')
    print(df.columns)
    # result = Chromaticity1.chord_chromaticity(chord=TonalHarmony.from_df_row())

    # result = df.apply(lambda row: Chromaticity1.chord_chromaticity(chord=TonalHarmony.from_df_row(df.iloc[row])))
    # [MyClass(row['A'], row['B']) for index, row in df.iterrows()]
    # [my_function(row) for index, row in df.iterrows()]
    result = Chromaticity1.df_chord_chromaticity(piece=df)
    print(f'{result=}')
