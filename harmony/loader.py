from __future__ import annotations
from harmony.representation import Key, Numeral
import os
from dataclasses import dataclass
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

import util


@dataclass
class AspectInfo:

    def n_grams(self, n: int, label_seq: List) -> np.ndarray:
        n_grams = util.get_n_grams(sequence=label_seq, n=n)
        return n_grams

    def transition_matrix(self, n: int, label_seq: List, probability: bool = True):
        n_grams = self.n_grams(n=n, label_seq=label_seq)
        transition_matrix = util.get_transition_matrix(n_grams=n_grams)
        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix

    def unique_labels(self, label_seq: List) -> List:
        unique_labels = list(set(label_seq))
        return unique_labels

    def remove_repeated_labels_occurrences(self, label_seq: List) -> List:
        """Transform label seq [A, A, A, B, C, C, A, C, C, C] to [A, B, C, A, C]"""
        prev = object()
        occurrence_list = [prev := v for v in label_seq if prev != v]
        return occurrence_list


@dataclass
class HarmonyInfo(AspectInfo):
    globalkey: Key
    localkey_seq: List[Numeral]
    chord_seq: List[str]
    numeral_seq: List[Numeral]
    chord_type_seq: List[str]
    root_seq: List[int]
    bass_note_seq: List[int]

    def localkey_labels_occurrences(self) -> List:
        localkey_labels_occurrences = self.remove_repeated_labels_occurrences(label_seq=self.localkey_seq)
        return localkey_labels_occurrences

    def modulation_bigrams_list(self) -> List[str]:
        """Returns a list of str representing the modulation bigram. e.g., "f#_IV/V_bIII/V" """
        globalkey = self.globalkey.to_str()
        localkey_list = self.localkey_labels_occurrences()
        mod_bigrams = util.get_n_grams(sequence=localkey_list, n=2)
        mod_bigrams = ["_".join([item[0], item[1]]) for item in mod_bigrams]
        bigrams = [globalkey + '_' + item for item in mod_bigrams]
        return bigrams


class MeasureInfo:
    pass


class NoteInfo:
    pass


@dataclass
class PieceMetaInfo:
    corpus_name: str
    composed_start: int
    composed_end: int
    composer: str
    annotated_key: Key
    label_count: int | None


@dataclass
class PieceInfo:
    # containing the data for a single piece
    meta_info: PieceMetaInfo
    harmony_info: HarmonyInfo
    measure_info: MeasureInfo
    note_info: NoteInfo

    @classmethod
    def from_corpus_directory(cls, parent_corpus_path: str, piece_name: str) -> PieceInfo:

        def read_tsv_data(source_df: pd.DataFrame, selected_col_name: str) -> List:
            """read a column of dataframe, and output it as a list"""
            result = source_df[selected_col_name].values.flatten().tolist()
            return result

        corpus_name: str = parent_corpus_path.split(os.sep)[-2]
        metadata_tsv_df: pd.DataFrame = pd.read_csv(parent_corpus_path + 'metadata.tsv', sep='\t')
        annotated_key = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['annotated_key']
        composer = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['composer']
        composed_start = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['composed_start']
        composed_end = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['composed_end']
        label_count = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == piece_name]['label_count']

        meta_info = PieceMetaInfo(
            corpus_name=corpus_name,
            composed_start=composed_start,
            composed_end=composed_end,
            composer=composer,
            annotated_key=Key.parse(key_str=annotated_key),
            label_count=label_count)

        harmoniees_df = pd.DataFrame = pd.read_csv(parent_corpus_path + 'harmonies/' + piece_name + '.tsv', sep='\t')

        localkey_seq = harmoniees_df['localkey'].values.flatten().tolist()
        chord_seq = harmoniees_df['chord'].values.flatten().tolist()
        numeral_seq = harmoniees_df['numeral'].values.flatten().tolist()
        chord_type_seq = harmoniees_df['chord_type'].values.flatten().tolist()
        root_seq = harmoniees_df['root'].values.flatten().tolist()
        bass_note_seq = harmoniees_df['bass_note'].values.flatten().tolist()

        harmony_info = HarmonyInfo(
            globalkey=annotated_key,
            localkey_seq=localkey_seq,
            chord_seq=chord_seq,
            numeral_seq=numeral_seq,
            chord_type_seq=chord_type_seq,
            root_seq=root_seq,
            bass_note_seq=bass_note_seq
        )
        measure_info = MeasureInfo()
        note_info = NoteInfo()

        instance = cls(
            meta_info=meta_info,
            harmony_info=harmony_info,
            measure_info=measure_info,
            note_info=note_info,
        )
        return instance

    # ===================================================

    def get_aspect_df(self, aspect: Literal['harmonies', 'measures', 'notes'],
                      selected_keys: Optional[List[str]]) -> pd.DataFrame:
        """
        To read the piecewise aspect(harmonies/measures/notes) tsv files as a DataFrame, always attach metadata (parent corpus, fname)
        :param: selected_keys: a list of keys (such as 'chord','numeral') in the tsv file. If none (default), select all keys.
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
            selected_df = all_df[selected_keys]
            selected_df = selected_df.dropna()  # to drop the rows with index NaN
            return selected_df

    def get_piecewise_unique_key_values(self, aspect: Literal['harmonies', 'measures', 'notes'],
                                        key: str) -> List[str]:
        if aspect == 'harmonies':
            df = self.get_aspect_df(aspect='harmonies', selected_keys=None)
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

        elif aspect == 'measures':
            df = self.get_aspect_df(aspect='measures', selected_keys=None)
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

        elif aspect == 'notes':
            df = self.get_aspect_df(aspect='notes', selected_keys=None)
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

    def get_n_grams(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'], key: str) -> np.ndarray:
        key_val_list = self.get_aspect_df(aspect=aspect, selected_keys=[key]).values.flatten().tolist()
        n_grams = util.get_n_grams(sequence=key_val_list, n=n)
        return n_grams

    def get_transition_matrix(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'],
                              key: str,
                              probability: bool = False) -> pd.DataFrame:
        n_grams = self.get_n_grams(n=n, aspect=aspect, key=key)
        transition_matrix = util.get_transition_matrix(n_grams=n_grams)
        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix

    def _get_composed_year(self) -> int:
        metadata_tsv_df = pd.read_csv(self.parent_corpus_path + 'metadata.tsv', sep='\t')
        composed_year_df = metadata_tsv_df.loc[metadata_tsv_df['fnames'] == self.piece_name]['composed_end']
        composed_year = composed_year_df.values[0]
        return composed_year

    def _global_mode(self) -> str:

        if 'globalkey_is_minor' in self.get_aspect_df(aspect='harmonies', selected_keys=None).columns:
            key = self.get_aspect_df(aspect='harmonies', selected_keys=['globalkey_is_minor'])[
                'globalkey_is_minor'].mode().values[0]
            if key == 0:
                return str('MAJOR')
            elif key == 1:
                return str('MINOR')
            else:
                raise ValueError
        else:
            return str('NA')

    def get_localkey_label_list(self) -> List[str]:
        """
        Get the list of harmony list (localkey), e.g. "['I', 'iii', 'I', 'IV', 'III', 'I']"
        """
        localkey_list = self.get_aspect_df(aspect='harmonies', selected_keys=['localkey']).values.flatten().tolist()
        prev = object()
        localkey_list = [prev := v for v in localkey_list if prev != v]
        return localkey_list

    def get_modulation_bigrams_list(self) -> List[str]:
        globalkey = self.get_aspect_df(aspect='harmonies', selected_keys=['globalkey']).values.flatten()[0]
        localkey_list = self.get_localkey_label_list()
        modulation_bigrams = util.get_n_grams(sequence=localkey_list, n=2)
        modulation_bigrams = ["_".join([item[0], item[1]]) for idx, item in enumerate(modulation_bigrams)]

        globalkey_modulation_bigrams = [globalkey + '_' + item for idx, item in enumerate(modulation_bigrams)]
        return globalkey_modulation_bigrams

    def get_key_region_subdfs_list(self) -> List[pd.DataFrame]:
        """
        Get a list of pd.Dataframe of different localkey region info.
        columns: ["globalkey", "localkey", "chord", "numeral", "form", "figbass", "changes", "relativeroot",
                "root", "bass_note", "key_region_label"]
        """
        harmonies_df = self.get_aspect_df(aspect='harmonies', selected_keys=None)

        harmonies_df['key_region_label'] = harmonies_df['localkey'].ne(harmonies_df['localkey'].shift()).cumsum()
        harmonies_df = harmonies_df.groupby('key_region_label')
        localkey_df = harmonies_df[
            ["globalkey", "localkey", "chord", "numeral", "form", "figbass", "changes", "relativeroot",
             "root", "bass_note", "key_region_label"]]
        dfs = []
        for name, data in localkey_df:
            dfs.append(data)

        return dfs


@dataclass
class CorpusInfo:
    # containing data for a single corpus
    corpus_path: str

    def __post_init__(self):
        self.corpus_name = self.corpus_path.split(os.sep)[-2]
        self.metadata_df = self._get_metadata()

        self.annotated_piece_name_list = self.get_annotated_piece_name_list(manually_filtered_pieces=None)
        self.annotated_piece_list: List[PieceInfo] = [PieceInfo(parent_corpus_path=self.corpus_path, piece_name=name)
                                                      for name in self.annotated_piece_name_list]

        self.harmonies_df = self.get_corpus_aspect_df(aspect='harmonies', selected_keys=None)

    def _get_metadata(self) -> pd.DataFrame:
        """
        read the meatadata.tsv into DataFrame, also add a column of 'corpus' to indicate corpus name
        """

        metadata_tsv_path = self.corpus_path + 'metadata.tsv'
        metadata_df = pd.read_csv(metadata_tsv_path, sep='\t')
        corpus_name_list = [self.corpus_name] * len(metadata_df)
        metadata_df['corpus'] = corpus_name_list
        # metadata_df['major/minor'] = metadata_df['annotated_key'].map(MAJOR_MINOR_KEYS_Dict)
        return metadata_df

    def get_annotated_piece_name_list(self, manually_filtered_pieces: Optional[List[str]] = None):
        """check if label count is 0 (not annotated) or not. """
        sub_df = self.metadata_df[['fnames', 'label_count']]
        df2check = sub_df.drop(sub_df[sub_df['label_count'] == 0].index)
        annotated_piece_name_list = df2check['fnames'].values.flatten().tolist()

        if manually_filtered_pieces:
            filtered_annotated_piece_name_list = [annotated_piece_name_list.remove(val) for idx, val in
                                                  enumerate(manually_filtered_pieces)]
            return filtered_annotated_piece_name_list
        else:
            return annotated_piece_name_list

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
        for idx, val in enumerate(self.annotated_piece_name_list):
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

            # aspect_all_df['globalkey'] = [aspect_all_df.globalkey] * df_length

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
            df = self.get_corpus_aspect_df(aspect='harmonies', selected_keys=None)
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

        elif aspect == 'measures':
            df = self.get_corpus_aspect_df(aspect='measures', selected_keys=None)
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

        elif aspect == 'notes':
            df = self.get_corpus_aspect_df(aspect='notes', selected_keys=None)
            unique_key_vals = df[key].unique().tolist()
            return unique_key_vals

    def get_corpus_localkey_bigrams(self):
        corpus_modulation_list = []
        for idx, val in enumerate(self.annotated_piece_name_list):
            piece = PieceInfo(parent_corpus_path=self.corpus_path, piece_name=val)
            piece_modulation_df = pd.DataFrame(piece.get_localkey_label_list(), columns=['modulations'])
            piece_modulation_df['fname'] = val
            piece_modulation_df['composed_end'] = \
                self.metadata_df[self.metadata_df['fnames'] == val]['composed_end'].values[0]
            piece_modulation_df['corpus'] = self.corpus_name
            corpus_modulation_list.append(piece_modulation_df)
        corpus_modulation_df = pd.concat(corpus_modulation_list, ignore_index=True)
        return corpus_modulation_df

    def is_annotated(self) -> bool:
        # whether the entire corpus is (at least partially) annotated or not.
        if all([os.path.isdir(self.corpus_path + 'harmonies/'),
                os.path.isdir(self.corpus_path + 'measures/'),
                os.path.isdir(self.corpus_path + 'notes/'), ]) is False:
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

        corpus_n_grams = []
        for piece in self.annotated_piece_name_list:
            key_values_list = piece.get_aspect_df(aspect=aspect, selected_keys=[key]).values.flatten().tolist()
            piece_n_grams = util.get_n_grams(sequence=key_values_list, n=n)
            corpus_n_grams.append(piece_n_grams)
        corpus_n_grams = np.concatenate(corpus_n_grams)

        return corpus_n_grams

    def get_transition_matrix(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'],
                              key: str,
                              select_top_ranking: Optional[int],
                              probability: bool = False) -> pd.DataFrame:
        if select_top_ranking:
            pass
        else:
            n_grams = self.get_n_grams(n=n, aspect=aspect, key=key)
            transition_matrix = util.get_transition_matrix(n_grams=n_grams)
            if probability:
                transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
                return transition_prob
            return transition_matrix


@dataclass
class MetaCorpraInfo:
    # containing data for a collection of corpora
    meta_corpora_path: str

    def __post_init__(self):
        self.corpus_name_list: List[str] = sorted([f for f in os.listdir(self.meta_corpora_path)
                                                   if not f.startswith('.')
                                                   if not f.startswith('__')])
        self.corpus_paths: List[str] = [self.meta_corpora_path + val + '/' for idx, val in
                                        enumerate(self.corpus_name_list)]
        self.corpus_list: List[CorpusInfo] = [CorpusInfo(corpus_path=path) for path in self.corpus_paths]
        # self.annotated_corpus_list: List[CorpusInfo] = [corpus_info for corpus_info in self.corpus_list if
        #                                                 corpus_info.is_annotated()]
        # self.harmonies_df = self.get_corpora_aspect_df(aspect='harmonies', selected_keys=None, annotated=True)

    def get_corpora_metadata_df(self, selected_keys: List[str]):
        metadata_list = []
        for corpus in self.corpus_list:
            corpus_metadata = corpus.metadata_df[corpus.metadata_df['label_count'] > 0]
            corpus_metadata = corpus_metadata[selected_keys]
            metadata_list.append(corpus_metadata)
        metadata_df = pd.concat(metadata_list, ignore_index=True)
        return metadata_df

    def get_corpora_aspect_df(self, aspect: Literal['harmonies', 'measures', 'notes'],
                              selected_keys: Optional[List[str]], annotated: bool = True) -> pd.DataFrame:
        if annotated is True:
            concat_df_list = []
            for idx, val in enumerate(self.corpus_list):
                aspect_all_df = val.get_corpus_aspect_df(aspect=aspect, selected_keys=selected_keys)
                concat_df_list.append(aspect_all_df)
            corpora_aspect_df = pd.concat(concat_df_list)
            # corpora_aspect_df = corpora_aspect_df.dropna()  # to drop the rows with index NaN
            return corpora_aspect_df
        else:
            concat_df_list = []
            for idx, val in enumerate(self.corpus_list):
                aspect_all_df = val.get_corpus_aspect_df(aspect=aspect, selected_keys=selected_keys)
                concat_df_list.append(aspect_all_df)
            corpora_aspect_df = pd.concat(concat_df_list)
            # corpora_aspect_df = corpora_aspect_df.dropna()  # to drop the rows with index NaN
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

    def get_top_ranking_labels(self, top_pos: int, rank_by: Literal['count'],
                               aspect: Literal['harmonies', 'measures', 'notes'], key: str) -> pd.DataFrame:

        if rank_by == 'count':
            label_list = self.get_corpora_aspect_df(aspect=aspect, selected_keys=[
                key]).value_counts().to_frame().reset_index()
            label_list.columns = [key, 'counts']

            top_ranking_labels = label_list.iloc[:top_pos]

            return top_ranking_labels
        else:
            raise NotImplementedError

    # def get_corpora_modulation_bigrams(self) -> pd.DataFrame:
    #     corpora_modulation_list = []
    #     for idx, val in enumerate(self.corpus_list):
    #         corpus_modulation_df = val.get_corpus_modulation_bigrams()
    #         corpora_modulation_list.append(corpus_modulation_df)
    #     corpora_modulation_df = pd.concat(corpora_modulation_list, ignore_index=True)
    #     return corpora_modulation_df

    def get_n_grams(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'],
                    key: str):

        metacorpora_n_grams = []
        for corpus in self.corpus_list:
            key_values_list = corpus.get_corpus_aspect_df(aspect=aspect,
                                                          selected_keys=[key]).values.flatten().tolist()
            corpus_n_grams = util.get_n_grams(sequence=key_values_list, n=n)
            metacorpora_n_grams.append(corpus_n_grams)

        metacorpora_n_grams = np.concatenate(metacorpora_n_grams)

        return metacorpora_n_grams

    def get_transition_matrix(self, n: int, aspect: Literal['harmonies', 'measures', 'notes'],
                              key: str,
                              probability: bool = False) -> pd.DataFrame:
        n_grams = self.get_n_grams(n=n, aspect=aspect, key=key)
        transition_matrix = util.get_transition_matrix(n_grams=n_grams)
        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix

    def get_corpus_list_in_chronological_order(self):
        corpus_year_df_list = []

        for corpus in self.corpus_list:
            mean_year = corpus.metadata_df['composed_end'].mean()
            mean_year = int(mean_year)
            corpuswise_corpus_year_df = pd.DataFrame([[corpus.corpus_name, mean_year]], columns=['corpus', 'year'])
            corpus_year_df_list.append(corpuswise_corpus_year_df)
        corpus_year_df = pd.concat(corpus_year_df_list)
        sorted_df = corpus_year_df.sort_values(by=['year'], ascending=True)
        corpus_list_in_chronological_order = sorted_df['corpus'].to_list()

        return corpus_list_in_chronological_order


if __name__ == '__main__':
    piece = PieceInfo(parent_corpus_path="../romantic_piano_corpus/debussy_suite_bergamasque/",
                      piece_name="l075-01_suite_prelude")

    # metacorpora = MetaCorpraInfo(meta_corpora_path='../petit_dcml_corpus/')
    # result = metacorpora.get_corpora_unique_key_values(aspect='harmonies', key='localkey')
    # print(result)

    # df = metacorpora.get_corpora_aspect_df('harmonies', selected_keys=['corpus', 'fname', 'composed_end', 'localkey'])

    df = piece.get_aspect_df('harmonies',
                             selected_keys=['corpus', 'fname', 'localkey'])

    one_hot_localkey_df = pd.get_dummies(df['localkey'])
    print(one_hot_localkey_df)
