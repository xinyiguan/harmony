"""
The loader contains three class to read the
"""
import fnmatch
import os
from dataclasses import dataclass
from typing import List, Optional, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from n_gram import get_n_grams, get_transition_matrix


@dataclass
class PieceInfo:
    # assuming we are inside the corpus folder
    parent_corpus_path: str
    piece_name: str

    def __post_init__(self):
        self.corpus_name: str = self.parent_corpus_path.split(os.sep)[-2]

    def get_aspect_df(self, aspect: Literal['harmonies', 'measures', 'notes'],
                      selected_keys: Optional[List[str]]) -> pd.DataFrame:
        """
        To get the piece-wise aspect(harmonies/measures/notes) tsv files as a DataFrame, always attach metadata (parent corpus, fname)
        :param: selected_keys: a list of keys (such as 'chord','numeral') in the tsv file. If none, return the entire df.
        :return: a DataFrame
        """

        piece_aspect_tsv_path = self.parent_corpus_path + aspect + '/' + self.piece_name + '.tsv'
        all_df = pd.read_csv(piece_aspect_tsv_path, sep='\t')
        df_length = all_df.shape[0]
        all_df['corpus'] = [self.corpus_name] * df_length
        all_df['fname'] = [self.piece_name] * df_length

        if selected_keys is None:
            return all_df
        else:
            selected_df = all_df[selected_keys].copy()
            selected_df = selected_df.dropna()  # to drop the rows with index NaN
            return selected_df

    def get_piecewise_unique_key_values(self, aspect: Literal['harmonies', 'measures', 'notes'],
                                        key: str) -> List[str]:
        if aspect == 'harmonies':
            df = self.harmonies_df
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

        elif aspect == 'measures':
            df = self.measures_df
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

        elif aspect == 'notes':
            df = self.notes_df
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

    def get_n_grams(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'], key: str) -> np.ndarray:
        key_val_list = self.get_aspect_df(aspect=aspect, selected_keys=[key]).values.flatten().tolist()
        n_grams = get_n_grams(sequence=key_val_list, n=n)
        return n_grams

    def get_transition_matrix(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'],
                              key: str,
                              probability: bool = False) -> pd.DataFrame:
        n_grams = self.get_n_grams(n=n, aspect=aspect, key=key)
        transition_matrix = get_transition_matrix(n_grams=n_grams)
        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix


@dataclass
class CorpusInfo:
    corpus_path: str

    def __post_init__(self):
        self.corpus_name = self.corpus_path.split(os.sep)[-2]
        self.piece_name_list = sorted(self._get_metadata()['fnames'])
        self.piece_list: List[PieceInfo] = [PieceInfo(parent_corpus_path=self.corpus_path, piece_name=name) for name in
                                            self.piece_name_list]
        self.metadata_df = self._get_metadata()

    def _get_metadata(self) -> pd.DataFrame:
        """
        read the meatadata.tsv into DataFrame, also add a column of 'corpus' to indicate corpus name
        """

        metadata_tsv_path = self.corpus_path + 'metadata.tsv'
        metadata_df = pd.read_csv(metadata_tsv_path, sep='\t')
        corpus_name_list = [self.corpus_name] * len(metadata_df)
        metadata_df['corpus'] = corpus_name_list

        return metadata_df

    def get_corpus_aspect_df(self, aspect: Literal['harmonies', 'measures', 'notes'],
                             selected_keys: Optional[List[str]]) -> pd.DataFrame:
        """
        To concat all the sub-pieces info dataframe into a large one.
        For each piece, always attach metadata:  annotated_key, ambitus, composed_end, composed_start
        :param selected_keys:
        :param aspect: Literal['harmonies', 'measures', 'notes']
        :return: DataFrame
        """

        concat_df_list = []
        for idx, val in enumerate(self.piece_name_list):
            piece = PieceInfo(parent_corpus_path=self.corpus_path, piece_name=val)
            aspect_all_df = piece.get_aspect_df(aspect=aspect, selected_keys=None)

            df_length = aspect_all_df.shape[0]

            # get the row index of metadata with matching fnames:
            correpond_piece_row_idx_in_metadata_df = \
                self.metadata_df.index[self.metadata_df['fnames'] == val].to_list()[0]

            aspect_all_df['annotated_key'] = [self.metadata_df['annotated_key'][
                                                  correpond_piece_row_idx_in_metadata_df]] * df_length
            aspect_all_df['ambitus'] = [self.metadata_df['ambitus'][correpond_piece_row_idx_in_metadata_df]] * df_length
            aspect_all_df['composed_end'] = [self.metadata_df['composed_end'][
                                                 correpond_piece_row_idx_in_metadata_df]] * df_length
            aspect_all_df['composed_start'] = [self.metadata_df['composed_start'][
                                                   correpond_piece_row_idx_in_metadata_df]] * df_length
            concat_df_list.append(aspect_all_df)
        concat_aspect_df = pd.concat(concat_df_list)

        if selected_keys is None:
            return concat_aspect_df
        else:
            selected_aspect_df = concat_aspect_df[selected_keys].copy()
            selected_aspect_df = selected_aspect_df.dropna()  # to drop the rows with index NaN
            return selected_aspect_df

    def get_corpuswise_unique_key_values(self, aspect: Literal['harmonies', 'measures', 'notes'],
                                         key: str) -> List[str]:
        if aspect == 'harmonies':
            df = self.corpus_harmonies_df
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

        elif aspect == 'measures':
            df = self.corpus_measures_df
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

        elif aspect == 'notes':
            df = self.corpus_notes_df
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

    def is_annotated(self) -> bool:
        conditions_1 = [
            os.path.isdir(self.corpus_path + 'harmonies/'),
            os.path.isdir(self.corpus_path + 'measures/'),
            os.path.isdir(self.corpus_path + 'notes/'),
        ]
        if all(conditions_1) is False:
            return False
        else:
            conditions_2 = [
                any(file for file in os.listdir(self.corpus_path + 'harmonies/') if file.endswith('.tsv')),
                any(file for file in os.listdir(self.corpus_path + 'measures/') if file.endswith('.tsv')),
                any(file for file in os.listdir(self.corpus_path + 'notes/') if file.endswith('.tsv')),
            ]
            is_annotated = all(conditions_2)
            return is_annotated

    def get_n_grams(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'], key: str):

        # pieces = self.piece_list
        corpus_n_grams = []
        for piece in self.piece_list:
            key_values_list = piece.get_aspect_df(aspect=aspect, selected_keys=[key]).values.flatten().tolist()
            piece_n_grams = get_n_grams(sequence=key_values_list, n=n)
            corpus_n_grams.append(piece_n_grams)
        corpus_n_grams = np.concatenate(corpus_n_grams)

        return corpus_n_grams

    def get_transition_matrix(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'],
                              key: str,
                              probability: bool = False) -> pd.DataFrame:
        n_grams = self.get_n_grams(n=n, aspect=aspect, key=key)
        transition_matrix = get_transition_matrix(n_grams=n_grams)
        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix


@dataclass
class MetaCorpraInfo:
    meta_corpora_path: str

    def __post_init__(self):
        self.corpus_name_list: List[str] = sorted([f for f in os.listdir(self.meta_corpora_path)
                                                   if not f.startswith('.')
                                                   if not f.startswith('__')])
        self.corpus_paths: List[str] = [self.meta_corpora_path + val + '/' for idx, val in
                                        enumerate(self.corpus_name_list)]
        self.corpus_list: List[CorpusInfo] = [CorpusInfo(corpus_path=path) for path in self.corpus_paths]
        self.annotated_corpus_list: List[CorpusInfo] = [corpus_info for corpus_info in self.corpus_list if
                                                        corpus_info.is_annotated()]

    def get_corpora_concat_metadata_df(self, selected_keys: List[str]):
        metadata_list = []
        for corpus in self.annotated_corpus_list:
            corpus_metadadta = corpus.metadata_df[selected_keys]
            metadata_list.append(corpus_metadadta)
        metadata_df = pd.concat(metadata_list, ignore_index=True)
        return metadata_df

    def get_corpora_aspect_df(self, aspect: Literal['harmonies', 'measures', 'notes'],
                              selected_keys: Optional[List[str]], annotated: bool = True) -> pd.DataFrame:
        if annotated is True:
            concat_df_list = []
            for idx, val in enumerate(self.annotated_corpus_list):
                aspect_all_df = val.get_corpus_aspect_df(aspect=aspect, selected_keys=selected_keys)
                concat_df_list.append(aspect_all_df)
            corpora_aspect_df = pd.concat(concat_df_list)
            corpora_aspect_df = corpora_aspect_df.dropna()  # to drop the rows with index NaN
            return corpora_aspect_df
        else:
            concat_df_list = []
            for idx, val in enumerate(self.corpus_list):
                aspect_all_df = val.get_corpus_aspect_df(aspect=aspect, selected_keys=selected_keys)
                concat_df_list.append(aspect_all_df)
            corpora_aspect_df = pd.concat(concat_df_list)
            corpora_aspect_df = corpora_aspect_df.dropna()  # to drop the rows with index NaN
            return corpora_aspect_df

    def get_corpora_unique_key_values(self, aspect: Literal['harmonies', 'measures', 'notes'],
                                      key: str, annotated: bool = True) -> List[str]:
        if annotated:
            if aspect == 'harmonies':
                df = self.get_corpora_aspect_df(aspect=aspect, selected_keys=None, annotated=True)
                unique_key_vals = df[key].unique().tolist()
                return unique_key_vals

            elif aspect == 'measures':
                df = self.get_corpora_aspect_df(aspect=aspect, selected_keys=None, annotated=True)
                unique_key_vals = df[key].unique().tolist()
                return unique_key_vals

            elif aspect == 'notes':
                df = self.get_corpora_aspect_df(aspect=aspect, selected_keys=None, annotated=True)
                unique_key_vals = df[key].unique().tolist()
                return unique_key_vals

    def get_n_grams(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'], key: str):

        metacorpora_n_grams = []
        for corpus in self.corpus_list:
            key_values_list = corpus.get_corpus_aspect_df(aspect=aspect, selected_keys=[key]).values.flatten().tolist()
            corpus_n_grams = get_n_grams(sequence=key_values_list, n=n)
            metacorpora_n_grams.append(corpus_n_grams)

        metacorpora_n_grams = np.concatenate(metacorpora_n_grams)

        return metacorpora_n_grams

    def get_transition_matrix(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'],
                              key: str,
                              probability: bool = False) -> pd.DataFrame:
        n_grams = self.get_n_grams(n=n, aspect=aspect, key=key)
        transition_matrix = get_transition_matrix(n_grams=n_grams)
        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix


MAJOR = Literal['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', '#I', '#II', '#III', '#IV', '#V', '#VI', '#VII',
                'bI', 'bII', 'bIII', 'bIV', 'bV', 'bVI', 'bVII']

MINOR = Literal['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', '#i', '#ii', '#iii', '#iv', '#v', '#vi', '#vii',
                'bi', 'bii', 'biii', 'biv', 'bv', 'bvi', 'bvii']

if __name__ == '__main__':
    metacorpora_path = 'dcml_corpora/'
    metacorpora = MetaCorpraInfo(metacorpora_path)
    #
    # metadata=metacorpora.get_corpora_metadata_df()['fnames', 'composed_end']
    # print(metadata)

    metadata=metacorpora.get_corpora_concat_metadata_df(selected_keys=['corpus', 'fnames', 'composed_end'])
    print(metadata['composed_end'].isnull().values.any())