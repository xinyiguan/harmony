from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List, Callable, Self

import numpy as np
import pandas as pd

from data_strcutures import PieceMetaData, HarmonyInfo, NoteInfo, MeasureInfo


@dataclass
class PieceInfo:
    # containing the data for a single piece
    meta_info: PieceMetaData
    harmony_info: HarmonyInfo
    measure_info: MeasureInfo
    note_info: NoteInfo

    @classmethod
    def from_directory(cls, parent_corpus_path: str, piece_name: str) -> Self:

        corpus_name: str = parent_corpus_path.split(os.sep)[-2]
        metadata_tsv_df: pd.DataFrame = pd.read_csv(parent_corpus_path + 'metadata.tsv', sep='\t')







@dataclass
class BaseCorpusInfo(ABC):
    @abstractmethod
    def filter_pieces_by_condition(self, condition: Callable[[PieceInfo, ], bool]) -> List[PieceInfo]:
        pass


@dataclass
class CorpusInfo(BaseCorpusInfo):
    raise NotImplementedError


@dataclass
class MetacorporaInfo:
    raise NotImplementedError
