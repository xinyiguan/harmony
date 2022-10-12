import typing
from dataclasses import dataclass
import os

import pandas as pd


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

    def get_metadata(self) -> pd.DataFrame:
        metadata_tsv_path = self.corpus_path + 'metadata.tsv'
        metadata = pd.read_csv(metadata_tsv_path, sep='\t')
        corpus_name = self.corpus_path.split(os.sep)[
            -2]  # ['romantic_piano_corpus', 'chopin_mazurkas', ''] --> get the -2
        corpus_name_list = [corpus_name] * len(metadata)
        metadata['corpus'] = corpus_name_list
        return metadata

    # def get_corpus_harmonies_df(self, attach_metadata: bool = False) -> pd.DataFrame:
    #
    #     harmonies_folder_path = self.corpus_path + 'harmonies/'
    #     harmonies_tsv_list = [f for f in os.listdir(harmonies_folder_path)]
    #     fnames_list = [f.replace('.tsv', '') for f in harmonies_tsv_list]
    #     full_harmonies_tsv_paths_list = [harmonies_folder_path + item for item in harmonies_tsv_list]
    #
    #     if attach_metadata:
    #         metadata_df = corpus.metadata[['fnames', 'composed_end', 'key','mode']] # attach these meta info into each
    #         metadata_df= metadata_df.set_index('fnames')
    #
    #         pd_list = []
    #         for idx, fname in enumerate(fnames_list):
    #             tsv_path = harmonies_folder_path+fname+'.tsv'
    #             piecewise_harmonies_df = pd.read_csv(tsv_path, sep='\t')
    #             df_length = piecewise_harmonies_df.shape[0]
    #
    #             if metadata_df['fnames']:
    #             selected_metadata_row= metadata_df.loc[fname]
    #
    #             metadata_columns_repeated=pd.concat([selected_metadata_row]*df_length)
    #
    #
    #             pd_list.append(piecewise_harmonies_df)
    #         corpus_harmonies_df = pd.concat(pd_list)
    #         return corpus_harmonies_df
    #
    #     else:
    #         pd_list = []
    #         for idx, tsv_path in enumerate(full_harmonies_tsv_paths_list):
    #             piecewise_harmonies_df = pd.read_csv(tsv_path, sep='\t')
    #             # piecewise_harmonies_df['composition_endyear']=composition_endyr[idx]
    #             pd_list.append(piecewise_harmonies_df)
    #         corpus_harmonies_df = pd.concat(pd_list)
    #         return corpus_harmonies_df

    # def get_piece_info(self) -> typing.Dict[str, PieceInfo]:
    #     """
    #     :return: a dictionary {fname: PieceInfo}
    #     """
    #     return_dict = {}
    #     fname_list = self.metadata['fnames']
    #     for idx, fname in fname_list:
    #         piece_info = PieceInfo(fname)
    #         return_dict[fname] = piece_info
    #     return return_dict

    def get_corpus_harmonies_df(self):
        harmonies_folder_path = self.corpus_path + 'harmonies/'
        harmonies_tsv_list = [f for f in os.listdir(harmonies_folder_path)]
        print(harmonies_tsv_list)
        full_harmonies_tsv_paths_list = [harmonies_folder_path + item for item in harmonies_tsv_list]
        pd_list = []
        for idx, tsv_path in enumerate(full_harmonies_tsv_paths_list):
            piecewise_harmonies_df = pd.read_csv(tsv_path, sep='\t')
            pd_list.append(piecewise_harmonies_df)
        corpus_harmonies_df = pd.concat(pd_list)
        return corpus_harmonies_df

@dataclass
class MetaCorporaInfo:
    metacorpora_path: str

    def __post_init__(self):
        self.subcorpus_list = [f for f in os.listdir(self.metacorpora_path) if not f.startswith('.') if
                               not f.startswith('__')]
        self.subcorpus_path = [self.metacorpora_path + item + '/' for item in self.subcorpus_list]
        self.metadata = self.get_metadata()

    def get_metadata(self, write_csv: bool = False) -> pd.DataFrame:
        df_list = []
        for idx, val in enumerate(self.subcorpus_path):
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

    def get_corpora_harmonies(self) -> pd.DataFrame:
        df_list = []
        for idx, val in enumerate(self.subcorpus_path):
            subcorpus = self.get_subcorpus(val)
            subcorpus_harmonies = subcorpus.corpus_harmonies_df
            df_list.append(subcorpus_harmonies)
        corpora_harmonies = pd.concat(df_list)
        return corpora_harmonies


if __name__ == '__main__':
    # metacorpora_path = 'romantic_piano_corpus/'
    # metacorpora = MetaCorporaInfo(metacorpora_path)
    # metadata = metacorpora.metadata
    # subcorpus = metacorpora.get_subcorpus('chopin_mazurkas/')
    # meta=subcorpus.metadata
    # print(meta)
    corpus_path = 'romantic_piano_corpus/chopin_mazurkas/'
    corpus = CorpusInfo(corpus_path=corpus_path)

    metadata_df = corpus.metadata[['fnames', 'composed_end', 'key','mode']] # attach these meta info into each
    # metadata_df= metadata_df.set_index('fnames')
    print(metadata_df)
    piecewise_harmonies_df = pd.read_csv(corpus_path+'harmonies/BI140.tsv', sep='\t')
    df_length = piecewise_harmonies_df.shape[0]

