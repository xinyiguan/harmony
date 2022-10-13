import typing
from dataclasses import dataclass
import os
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipyentropy


class PieceInfo:
    fname: str

    def __init__(self):
        self.harmonies_df = self.get_piecewise_harmonies_df()

    def get_piecewise_harmonies_df(self) -> pd.DataFrame:
        tsv_file_path = self.fname + '.tsv'
        all_df = pd.read_csv(tsv_file_path + '.tsv', sep='\t')
        return all_df


@dataclass
class CorpusInfo:
    corpus_path: str

    def __post_init__(self):
        self.metadata = self.get_metadata()
        self.corpus_harmonies_df = self.get_corpus_harmonies_df()
        self.corpus_harmonies_meta_df = self.get_corpus_harmonies_df(attach_metadata=True)

    def get_metadata(self) -> pd.DataFrame:
        metadata_tsv_path = self.corpus_path + 'metadata.tsv'
        metadata = pd.read_csv(metadata_tsv_path, sep='\t')
        corpus_name = self.corpus_path.split(os.sep)[
            -2]  # ['romantic_piano_corpus', 'chopin_mazurkas', ''] --> get the -2
        corpus_name_list = [corpus_name] * len(metadata)
        metadata['corpus'] = corpus_name_list
        return metadata

    def get_corpus_harmonies_df(self, attach_metadata: bool = False) -> pd.DataFrame:

        harmonies_folder_path = self.corpus_path + 'harmonies/'
        harmonies_tsv_list = [f for f in os.listdir(harmonies_folder_path)]
        fnames_list = [f.replace('.tsv', '') for f in harmonies_tsv_list]
        full_harmonies_tsv_paths_list = [harmonies_folder_path + item for item in harmonies_tsv_list]

        if attach_metadata:
            metadata_df = self.metadata[
                ['corpus', 'fnames', 'composed_end', 'annotated_key']]  # attach these meta info into each

            pd_list = []
            for idx, fname_tsv in enumerate(harmonies_tsv_list):
                if fname_tsv.endswith('tsv'):
                    fname = fname_tsv.replace('.tsv', '')
                    tsv_path = harmonies_folder_path + fname + '.tsv'
                    piecewise_harmonies_df = pd.read_csv(tsv_path, sep='\t')
                    df_length = piecewise_harmonies_df.shape[0]

                    selected_row_index = int(metadata_df[metadata_df['fnames'] == fname].index.values)
                    selected_metadata_row = metadata_df.iloc[selected_row_index].to_frame().T

                    metadata_columns_repeated = pd.DataFrame(np.repeat(selected_metadata_row.values, df_length, axis=0))
                    metadata_columns_repeated.columns = selected_metadata_row.columns
                    extended_df = pd.concat([piecewise_harmonies_df, metadata_columns_repeated], axis=1)

                    pd_list.append(extended_df)
                else:
                    pass
                corpus_harmonies_df = pd.concat(pd_list)
                return corpus_harmonies_df

        else:
            pd_list = []
            for idx, fname_tsv in enumerate(harmonies_tsv_list):
                if fname_tsv.endswith('tsv'):
                    fname = fname_tsv.replace('.tsv', '')
                    tsv_path = harmonies_folder_path + fname + '.tsv'
                    piecewise_harmonies_df = pd.read_csv(tsv_path, sep='\t')
                    pd_list.append(piecewise_harmonies_df)
                else:
                    pass
                corpus_harmonies_df = pd.concat(pd_list)
                return corpus_harmonies_df

    def get_harmonies_data_by_key(self, key: str) -> pd.DataFrame:
        data = self.corpus_harmonies_meta_df[key]
        return data

    def compute_probs(self, key: str) -> pd.DataFrame:
        data_df = self.corpus_harmonies_df[key]
        data_count = data_df.value_counts()
        probs = data_count / data_count.sum()
        return probs

    def compute_ic(self, key: str) -> pd.DataFrame:
        probs = self.compute_probs(key=key)
        h = -np.log2(probs)
        return h

    def compute_entropy(self, key: str) -> float:
        probs = self.compute_probs(key=key)
        H = -sum([p * np.log2(p) for idx, p in enumerate(probs)])
        return H

    def compute_entropy_scipy(self, key: str, base: int = 2) -> float:
        """Compute the entropy of a distribution, using scipy library"""
        probs = self.compute_probs(key=key)
        H = scipyentropy(probs, base=base)
        return H


@dataclass
class MetaCorporaInfo:
    metacorpora_path: str

    def __post_init__(self):
        """default is annotated subcorpus"""
        self.unannotated_subcorpus_list = [f for f in os.listdir(self.metacorpora_path) if not f.startswith('.') if
                                           not f.startswith('__')]
        self.unannotated_subcorpus_paths = [self.metacorpora_path + item + '/' for item in
                                            self.unannotated_subcorpus_list]

        self.subcorpus_list = self.get_annotated_subcorpus_list()
        self.subcorpus_paths = [self.metacorpora_path + item + '/' for item in self.subcorpus_list]
        self.metadata = self.get_metadata()
        self.corpora_harmonies_df = self.get_corpora_harmonies_df()
        self.corpora_harmonies_meta_df = self.get_corpora_harmonies_meta()

    def get_annotated_subcorpus_list(self):
        """check harmonies folder exist, check harmonies/*.tsv path exist"""

        annotated_subcorpus_list = []
        for idx, val in enumerate(self.unannotated_subcorpus_list):
            sample_tsv_parent_path = self.metacorpora_path + val + '/harmonies/'
            if os.path.exists(sample_tsv_parent_path):
                if any(fname.endswith('.tsv') for fname in os.listdir(sample_tsv_parent_path)):
                    annotated_subcorpus_list.append(val)
        return annotated_subcorpus_list

    def get_metadata(self, write_csv: bool = False) -> pd.DataFrame:
        df_list = []
        for idx, val in enumerate(self.subcorpus_paths):
            subcorpus = CorpusInfo(val)
            subcorpus_metadata = subcorpus.metadata
            df_list.append(subcorpus_metadata)
        corpora_metadata = pd.concat(df_list)

        if write_csv is True:
            corpora_metadata.to_csv('dataset_metadata.csv', sep='\t', index=False, index_label=True, header=True)

        return corpora_metadata

    def get_subcorpus(self, subcorpus_path: str) -> CorpusInfo:
        subcorpus = CorpusInfo(subcorpus_path)
        return subcorpus

    def get_corpora_harmonies_df(self) -> pd.DataFrame:
        df_list = []
        for idx, val in enumerate(self.subcorpus_paths):
            subcorpus = self.get_subcorpus(val)
            subcorpus_harmonies = subcorpus.corpus_harmonies_df
            df_list.append(subcorpus_harmonies)
        corpora_harmonies = pd.concat(df_list)
        return corpora_harmonies

    def get_corpora_harmonies_meta(self) -> pd.DataFrame:
        df_list = []
        for idx, val in enumerate(self.subcorpus_paths):
            subcorpus = self.get_subcorpus(val)
            subcorpus_harmonies = subcorpus.corpus_harmonies_meta_df
            df_list.append(subcorpus_harmonies)
        corpora_harmonies = pd.concat(df_list)
        return corpora_harmonies

    def get_harmonies_data_by_key(self, key: str) -> pd.DataFrame:
        data = self.corpora_harmonies_meta_df[key]
        return data

    def get_vals_by_key(self, key: str) -> typing.List[str]:
        vals = self.corpora_harmonies_meta_df[key].to_list()
        return vals

    def get_unique_values_by_key(self, key: str) -> typing.List[str]:

        metacorpora_keyval_list = self.get_vals_by_key(key=key)

        metacorpora_keyval_list = [str(item) for item in metacorpora_keyval_list]
        metacorpora_keyval_list[:] = [value for value in metacorpora_keyval_list if value != 'nan']

        unique_values_in_key = [str(item) for item in metacorpora_keyval_list]
        unique_values_in_key = sorted(list(set(unique_values_in_key)))
        unique_values_in_key[:] = [value for value in unique_values_in_key if
                                   value != 'nan']  # get rid of 'nan' in the numeral list
        return unique_values_in_key

    def get_extended_df_by_key_stats(self, key: str, stats: str) -> pd.DataFrame:
        unique_key_vals = metacorpora.get_unique_values_by_key(key=key)
        key_df_list = []
        for idx, subcorpus_name in enumerate(self.subcorpus_list):
            subcorpus_path = metacorpora_path + subcorpus_name + '/'
            subcorpus = metacorpora.get_subcorpus(subcorpus_path)
            keyvalues_probs = subcorpus.compute_probs(key=key)
            extended_keyvalues_probs = pd.DataFrame(data=keyvalues_probs, index=unique_key_vals)
            current_col_name = extended_keyvalues_probs.columns.values.tolist()[0]
            extended_keyvalues_probs = extended_keyvalues_probs.rename(
                columns={current_col_name: subcorpus_name})  # rename the column

            key_df_list.append(extended_keyvalues_probs)
        concat_df = pd.concat(key_df_list, axis=1)
        return concat_df

    def get_sorted_key_value_list(self, key: str, method: typing.Literal['count', 'fifth']):
        if method == 'count':
            keyval_counts_df = self.corpora_harmonies_meta_df[key].value_counts().to_frame()
            sorted_key_val_list = keyval_counts_df.index.values.tolist()
            return sorted_key_val_list

        elif method == 'fifth':
            raise NotImplementedError

    def get_sorted_subcorpus_by_year(self, output_format: typing.Literal['year_corpus_df', 'corpus', 'years'])-> typing.Union[np.array, pd.DataFrame]:
        mean_comp_years = []
        for idx, val in enumerate(self.subcorpus_list):
            subcorpus = CorpusInfo(self.metacorpora_path + val + '/')
            subcorpus_composition_year_list = subcorpus.metadata['composed_end']
            mean_composition_year = int(np.mean(subcorpus_composition_year_list))
            mean_comp_years.append(mean_composition_year)

        df = pd.DataFrame(self.subcorpus_list, mean_comp_years).sort_index(axis=0)
        if output_format == 'year_corpus_df':
            return df

        elif output_format =='corpus':
            return df.values.flatten()

        elif output_format =='years':
            return df.index.values()

if __name__ == '__main__':
    metacorpora_path = 'romantic_piano_corpus/'
    metacorpora = MetaCorporaInfo(metacorpora_path)

    # # vals_list = metacorpora.get_vals_by_key(key='numeral')
    # unique_key_vals = metacorpora.get_unique_values_by_key(key='numeral')
    #
    # # print(meta)
    # corpus_path = 'romantic_piano_corpus/chopin_mazurkas/'
    # corpus = CorpusInfo(corpus_path=corpus_path)
    # prob = corpus.compute_probs(key='numeral')
    #
    # chordvocab_probs = corpus.compute_probs(key='numeral')
    # extended_chordvocab_probs = pd.DataFrame(data=chordvocab_probs, index=unique_key_vals)
    # col_name = extended_chordvocab_probs.columns.values.tolist()[0]
    # print(col_name)
