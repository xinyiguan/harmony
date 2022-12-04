from __future__ import annotations
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

import util
from harmony.representation import Key, Numeral


@dataclass(frozen=True)
class TabularData(ABC):
    _df: pd.DataFrame

    @classmethod
    def from_pd_df(cls, df: pd.DataFrame):
        instance = cls(_df=df)
        return instance

    def get_aspect(self, key: str) -> SequentialData:
        series: pd.Series = self._df[key]
        sequential_data = SequentialData.from_pd_series(series)
        return sequential_data


@dataclass(frozen=True)
class SequentialData(ABC):
    _series: pd.Series

    @classmethod
    def from_pd_series(cls, series: pd.Series):
        instance = cls(_series=series)
        return instance

    def n_grams(self, n: int) -> np.ndarray:
        n_grams = util.get_n_grams(sequence=self._series, n=n)
        return n_grams

    def transition_matrix(self, n: int = 2, probability: bool = True) -> pd.DataFrame:
        n_grams = self.n_grams(n=n)
        transition_matrix = util.get_transition_matrix(n_grams=n_grams)
        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix

    def unique_labels(self, label_seq: List) -> List:
        unique_labels = list(set(label_seq))
        return unique_labels

    def get_changes(self) -> SequentialData:
        """Transform label seq [A, A, A, B, C, C, A, C, C, C] --->  [A, B, C, A, C]"""
        prev = object()
        occurrence_list = [prev := v for v in self._series if prev != v]
        series = pd.Series(occurrence_list)
        sequential_data = SequentialData.from_pd_series(series)
        return sequential_data

    def count(self):
        """Count the occurrences of objects in the sequence"""
        value_count = pd.value_counts(self._series)
        return value_count

    def probability(self) -> pd.Series:
        count = self.count()
        prob = count / count.sum()
        return prob

    def distribution_entropy(self) -> float:
        mean_entropy = self.event_entropy().mean()
        return mean_entropy

    def event_entropy(self) -> pd.Series:
        event_entropy = self.probability()
        array = -np.log(event_entropy)
        series = pd.Series(data=array, name='event_entropy')
        return series

    def information_content(self):
        raise NotImplementedError


# _____________________________ AspectInfo ______________________________________

@dataclass(frozen=True)
class HarmonyInfo(TabularData):

    def modulation_bigrams_list(self) -> List[str]:
        """Returns a list of str representing the modulation bigram. e.g., "f#_IV/V_bIII/V" """
        globalkey = self.harmony_df['globalkey'][0]
        localkey_label_seq = self.harmony_df['localkey']
        localkey_list = self.remove_repeated_labels_occurrences(label_seq=localkey_label_seq)
        mod_bigrams = util.get_n_grams(sequence=localkey_list, n=2)
        mod_bigrams = ["_".join([item[0], item[1]]) for item in mod_bigrams]
        bigrams = [globalkey + '_' + item for item in mod_bigrams]
        return bigrams

    def attribute_df(self, attribute_label_seq: List) -> pd.DataFrame:
        non_repeated_label_seq = self.remove_repeated_labels_occurrences(label_seq=attribute_label_seq)
        df = pd.DataFrame(non_repeated_label_seq, columns=['attribute'])
        return df


@dataclass(frozen=True)
class MeasureInfo(TabularData):
    pass


@dataclass(frozen=True)
class NoteInfo(TabularData):

    @cached_property
    def tpc(self) -> SequentialData:
        series = self._df['tpc']
        sequential = SequentialData.from_pd_series(series=series)
        return sequential


@dataclass(frozen=True)
class KeyInfo(TabularData):
    @cached_property
    def global_key(self) -> Key:
        key_str = self._df['globalkey'][0]
        key = Key.parse(key_str=key_str)
        return key

    @cached_property
    def local_key(self) -> SequentialData:
        local_key_series = self._df['localkey']
        sequential_data = SequentialData.from_pd_series(series=local_key_series)
        return sequential_data


# _____________________________ LevelInfo ______________________________________

@dataclass
class PieceMetaInfo:
    corpus_name: str
    piece_name: str
    composed_start: int
    composed_end: int
    composer: str
    annotated_key: str
    label_count: int | None


@dataclass
class PieceInfo:
    # containing the data for a single piece
    meta_info: PieceMetaInfo
    harmony_info: HarmonyInfo
    measure_info: MeasureInfo
    note_info: NoteInfo
    key_info: KeyInfo

    def __post_init__(self):
        self.annotated = self.annotated()

    @classmethod
    def from_directory(cls, parent_corpus_path: str, piece_name: str) -> PieceInfo:

        corpus_name: str = parent_corpus_path.split(os.sep)[-2]
        metadata_tsv_df: pd.DataFrame = pd.read_csv(parent_corpus_path + 'metadata.tsv', sep='\t')
        annotated_key: str = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['annotated_key'].values[0]
        composer: str = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['composer'].values[0]
        composed_start = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['composed_start'].values[0]
        composed_end = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['composed_end'].values[0]
        label_count = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['label_count'].values[0]

        meta_info = PieceMetaInfo(
            corpus_name=corpus_name,
            piece_name=piece_name,
            composed_start=composed_start,
            composed_end=composed_end,
            composer=composer,
            annotated_key=annotated_key,
            label_count=label_count)

        try:
            harmonies_df: pd.DataFrame = pd.read_csv(parent_corpus_path + 'harmonies/' + piece_name + '.tsv', sep='\t')
            measure_df: pd.DataFrame = pd.read_csv(parent_corpus_path + 'measures/' + piece_name + '.tsv', sep='\t')
            note_df: pd.DataFrame = pd.read_csv(parent_corpus_path + 'notes/' + piece_name + '.tsv', sep='\t')
        except:
            raise Warning('piece does not have all the required .tsv files in this corpus')

        harmony_info = HarmonyInfo.from_pd_df(df=harmonies_df)
        measure_info = MeasureInfo.from_pd_df(df=measure_df)
        note_info = NoteInfo.from_pd_df(df=note_df)

        key_df: pd.DataFrame = harmony_info._df[['globalkey', 'localkey']]
        key_info = KeyInfo.from_pd_df(df=key_df)

        instance = cls(
            meta_info=meta_info,
            harmony_info=harmony_info,
            measure_info=measure_info,
            note_info=note_info,
            key_info=key_info,
        )
        return instance

    def annotated(self) -> bool:
        label_count = self.meta_info.label_count
        if label_count is not None:
            return True
        else:
            return False


@dataclass
class CorpusMetaInfo:
    corpus_name: str
    composer: str
    piecename_list: List[str]


@dataclass
class CorpusInfo:
    # containing data for a single corpus
    meta_info: CorpusMetaInfo
    pieceinfo_list: List[PieceInfo]

    def __post_init__(self):
        self.annotated_pieceinfo_list = self.annotated_pieces()

    @classmethod
    def from_directory(cls, corpus_path: str) -> CorpusInfo:
        corpus_name: str = corpus_path.split(os.sep)[-2]
        metadata_tsv_df: pd.DataFrame = pd.read_csv(corpus_path + 'metadata.tsv', sep='\t')
        piecename_list = metadata_tsv_df['fnames'].tolist()

        pieceinfo_list = [PieceInfo.from_directory(parent_corpus_path=corpus_path, piece_name=item) for item in
                          piecename_list]
        meta_info = CorpusMetaInfo(
            corpus_name=corpus_name,
            composer=metadata_tsv_df['composer'].tolist()[0],
            piecename_list=piecename_list)

        instance = cls(meta_info=meta_info, pieceinfo_list=pieceinfo_list)
        return instance

    def annotated_pieces(self) -> List[PieceInfo]:
        annotated_pieces = [item for item in self.pieceinfo_list if item.annotated]
        return annotated_pieces


@dataclass
class MetaCorporaMetaInfo:
    corpusname_list: List[str]


@dataclass
class MetaCorporaInfo:
    # containing data for a collection corpora
    meta_info: MetaCorporaMetaInfo
    corpusinfo_list: List[CorpusInfo]

    @classmethod
    def from_directory(cls, metacorpora_path: str) -> MetaCorporaInfo:
        corpusname_list = sorted([f for f in os.listdir(metacorpora_path)
                                  if not f.startswith('.')
                                  if not f.startswith('__')])

        meta_info = MetaCorporaMetaInfo(corpusname_list=corpusname_list)

        corpusinfo_list = [CorpusInfo.from_directory(corpus_path=metacorpora_path + item + '/') for item in
                           corpusname_list]

        instance = cls(meta_info=meta_info, corpusinfo_list=corpusinfo_list)
        return instance


if __name__ == '__main__':
    corpus_info = CorpusInfo.from_directory(corpus_path='../romantic_piano_corpus/debussy_suite_bergamasque/')
