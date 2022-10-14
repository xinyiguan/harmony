"""
The loader contains three class to read the
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class PieceInfo:
    # assuming we are inside the corpus folder
    parent_corpus_path:str
    piece_name: str

    def __post_init__(self):
        self.harmonies_df = self.get_harmonies_df(selected_keys=None)
        self.measures_df = self.get_measures_df(selected_keys=None)
        self.notes_df = self.get_notes_df(selected_keys=None)

    def get_harmonies_df(self, selected_keys: Optional[List[str]]) -> pd.DataFrame:
        """
        To get the piece-wise harmonies tsv file as a DataFrame.
        :param selected_keys: a list of keys (such as 'chord','numeral') in the tsv file. If none, return the entire df.
        :return:
        """
        piece_harmonies_tsv_path = self.parent_corpus_path+'harmonies/' + self.piece_name + '.tsv'
        all_df = pd.read_csv(piece_harmonies_tsv_path, sep='\t')

        if selected_keys is None:
            return all_df
        else:
            selected_df = all_df[selected_keys]
            return selected_df

    def get_measures_df(self, selected_keys: Optional[List[str]]) -> pd.DataFrame:
        """
        To get the piece-wise measures tsv file as a DataFrame.
        :param selected_keys: a list of keys in the tsv file. If none, return the entire df.
        :return:
        """
        piece_measure_tsv_path = self.parent_corpus_path+'meausres/' + self.piece_name + '.tsv'
        all_df = pd.read_csv(piece_measure_tsv_path, sep='\t')

        if selected_keys is None:
            return all_df
        else:
            selected_df = all_df[selected_keys]
            return selected_df

    def get_notes_df(self, selected_keys: Optional[List[str]]) -> pd.DataFrame:
        """
        To get the piece-wise harmonies tsv file as a DataFrame.
        :param selected_keys: a list of keys (such as 'chord','numeral') in the tsv file. If none, return the entire df.
        :return:
        """
        piece_notes_tsv_path = self.parent_corpus_path+'notes/' + self.piece_name + '.tsv'
        all_df = pd.read_csv(piece_notes_tsv_path, sep='\t')

        if selected_keys is None:
            return all_df
        else:
            selected_df = all_df[selected_keys]
            return selected_df



@dataclass
class CorpusInfo:
    corpus_path: str

    def __post_init__(self):
        self.metadata_df = self._get_metadata()
        self.concat_harmonies_df = self._get_concat_harmonies_df()



    def _get_metadata(self) -> pd.DataFrame:
        """
        read the meatadata.tsv into DataFrame, also add a column of 'corpus' to indicate corpus name
        """

        metadata_tsv_path = self.corpus_path + 'metadata.tsv'
        metadata_df = pd.read_csv(metadata_tsv_path, sep='\t')
        corpus_name = self.corpus_path.split(os.sep)[-2]  # ['dcml_corpora', 'debussy_suite_bergamasque', '']
        corpus_name_list = [corpus_name] * len(metadata_df)
        metadata_df['corpus'] = corpus_name_list

        return metadata_df

    def _get_concat_harmonies_df(self):
        raise NotImplementedError



@dataclass
class MetaCorpraInfo:
    meta_corpora_path: str

    def __post_init__(self):
        self.corpus_list = [f for f in os.listdir(self.meta_corpora_path)
                            if not f.startswith('.')
                            if not f.startswith('__')]
        self.corpus_paths = [self.meta_corpora_path + val + '/' for idx, val in enumerate(self.corpus_list)]
        self.annotated_corpus_list = self._get_annotated_corpus_list()
        self.annotated_corpus_paths = [self.meta_corpora_path + val + '/' for idx, val in
                                       enumerate(self.annotated_corpus_list)]

    def _get_annotated_corpus_list(self) -> List[str]:
        """
        a band-aid method to get the annotated corpus list
        (check if tsv files in harmonies/ and measures/ and notes/ exist)
        """

        raise NotImplementedError

    def get_corpora_harmonies_df(self, contain_metadata: bool = True):
        raise NotImplementedError

    def get_corpora_measures_df(self, contain_metadata: bool = True):
        raise NotImplementedError

    def get_corpora_notes_df(self, contain_metadata: bool = True):
        raise NotImplementedError


if __name__ == '__main__':
    path = 'dcml_corpora/debussy_suite_bergamasque/'
    piece = PieceInfo(parent_corpus_path=path, piece_name='l075-01_suite_prelude')
    print(piece.get_harmonies_df(selected_keys=['chord', 'numeral']))
