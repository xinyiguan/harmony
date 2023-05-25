# Contains different "aspects" of corpus data: metadata, harmony data, note data, measure data, ...

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List, Callable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequentialData(ABC):
    _series: pd.Series


# =====================================================================================================================
@dataclass
class PieceMetaData:
    corpus_path: Optional[str]
    corpus_name: Optional[str]
    piece_name: Optional[str]
    composed_start: Optional[int]
    composed_end: Optional[int]
    composer: Optional[str]
    annotated_key: Optional[str]
    label_count: Optional[int]
    piece_length: int

    @cached_property
    def era(self) -> str:
        def determine_era_based_on_year(year) -> str:
            if 0 < year < 1650:
                return 'Renaissance'

            elif 1649 < year < 1759:
                return 'Baroque'

            elif 1758 < year < 1819:
                return 'Classical'

            elif 1819 < year < 1931:
                return 'Romantic'

        if self.composed_end is not None:
            return determine_era_based_on_year(year=self.composed_end)
        else:
            print(f'No composition year available in dataset!')


@dataclass
class CorpusMetaData:
    corpus_name: SequentialData
    composer: SequentialData
    composed_start: SequentialData
    composed_end: SequentialData
    annotated_key: SequentialData
    piecename_list: List[str]  # don't count pieces with label_count=0


@dataclass
class MetaCorporaMetaData:
    corpora_names: SequentialData
    composer: SequentialData
    composed_start: SequentialData
    composed_end: SequentialData
    annotated_key: SequentialData
    corpusname_list: List[str]


# =======

@dataclass
class HarmonyInfo:
    raise NotImplementedError


@dataclass
class MeasureInfo:
    raise NotImplementedError

@dataclass
class NoteInfo:
    raise NotImplementedError
