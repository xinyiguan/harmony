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
class Chromaticity1:
    """
    Adopt the metric that non-diatonic notes in the chords and sum the line of fifth values.
    """
    @staticmethod
    def chord_chromaticity(chord: TonalHarmony) -> typing.Dict[str, int]:
        """refernece tonics (3 levels): global tonic, local tonic and pseduo tonic"""
        pcs = chord.pcs()
        GT = NonDiatonicPCs(pcs=pcs, reference_key=chord.globalkey).sum_in_fifths
        LT = NonDiatonicPCs(pcs=pcs, reference_key=chord.globalkey).sum_in_fifths
        result_dict = {'GT': GT,
                       'LT': LT}
        return result_dict





if __name__ == '__main__':
    piece = PieceInfo.from_directory(piece_name='l075-01_suite_prelude',
                                     parent_corpus_path='/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/')

    result = Chromaticity1.chord_chromaticity(chord=TonalHarmony.parse_dcml(globalkey_str='C', localkey_numeral_str='iii', chord_str='V'))
    print(f'{result=}')