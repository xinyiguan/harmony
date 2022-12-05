# Created by Xinyi Guan in 2022.
import glob
import os
import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt

from harmony.loader import MetaCorporaInfo, TabularData, PieceInfo, CorpusInfo
import harmony.util as util


# 1. chord transitions and chord triplets (3-grams) and their information-theoretic properties

# 1.  entropy of local key (target key of modulation)

def assemble_localkey_df(metacorpora_path: str) -> pd.DataFrame:
    """A dataframe with cols: localkey, year, composer, era"""

    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    localkey_series = metacorpora.key_info.local_key._series
    year_series = metacorpora.meta_info.composed_end._series
    composer_series = metacorpora.meta_info.composer._series

    frame = {'localkey': localkey_series, 'year': year_series, 'composer': composer_series}

    localkey_df = pd.DataFrame(frame)
    localkey_df['era'] = localkey_df['year'].apply(lambda x: util.determine_era_based_on_year(x))

    return localkey_df


def assemble_piece_localkey_entropy_df(metacorpora_path: str):
    """
    A dataframe of piece entropy: piecename, entropy, corpus, year, era
    The entropy of unique local key labels in a piece
    """

    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    entropy_df_list = []
    for corpusinfo in metacorpora.meta_info.corpusinfo_list:
        piece_names = pd.Series([item for item in corpusinfo.meta_info.piecename_list], name='piece')
        piece_localkey_entropy = pd.Series(
            [item.key_info.local_key.unique_labels().entropy() for item in corpusinfo.meta_info.pieceinfo_list],
            name='entropy')
        piece_year = pd.Series(
            [item.meta_info.composed_end._series.values[0] for item in corpusinfo.meta_info.pieceinfo_list],
            name='year')
        piece_parent_corpus = pd.Series(
            [item.meta_info.corpus_name._series.values[0] for item in corpusinfo.meta_info.pieceinfo_list],
            name='corpus')

        frame = {'piece': piece_names,
                 'entropy': piece_localkey_entropy,
                 'year': piece_year,
                 'corpus': piece_parent_corpus
                 }

        localkey_entropy_df = pd.DataFrame(frame)
        entropy_df_list.append(localkey_entropy_df)
    entropy_df = pd.concat(entropy_df_list)
    entropy_df['era'] = entropy_df['year'].apply(lambda x: util.determine_era_based_on_year(x))
    return entropy_df


def plot_localkey_entropy_by_pieces(metacorpora_path: str, fig_path: str | None, savefig: bool = True):
    data = assemble_piece_localkey_entropy_df(metacorpora_path=metacorpora_path)
    sorted_df = data.sort_values(by=['year'])
    corpus_chronological_order = sorted_df['corpus'].unique().tolist()

    fig, ax = plt.subplots(figsize=(22, 12))
    sns.scatterplot(data=data, x='year', y='entropy', hue='corpus', hue_order=corpus_chronological_order)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)

    plt.title("Entropy of local keys in each piece")

    if savefig:
        if fig_path is None:
            fig_path = 'monday-inforamtion-theoretic-quantity-figs/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        plt.savefig(fname=fig_path + 'localkey_piece_entropy.jpeg', dpi=200, format='jpeg')
    return fig


def assemble_corpus_localkey_entropy_df(metacorpora_path: str):
    """A dataframe of corpus entropy: corpus, entropy, year, era"""
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    entropy_df_list = []
    for corpusinfo in metacorpora.meta_info.corpusinfo_list:
        corpus_localkey_entropy = pd.Series(corpusinfo.key_info.local_key.unique_labels().entropy(), name='entropy')
        corpus_name = pd.Series(corpusinfo.meta_info.corpus_name._series.values[0], name='corpus')
        corpus_year = pd.Series(int(corpusinfo.meta_info.composed_end.mean()), name='year')

        frame = {'entropy': corpus_localkey_entropy,
                 'corpus': corpus_name,
                 'year': corpus_year,
                 }
        corpus_entropy_df = pd.DataFrame(frame)
        entropy_df_list.append(corpus_entropy_df)
    entropy_df = pd.concat(entropy_df_list)
    entropy_df['era'] = entropy_df['year'].apply(lambda x: util.determine_era_based_on_year(x))
    return entropy_df


def plot_localkey_entropy_by_corpus(metacorpora_path: str, fig_path: str | None, savefig: bool = True):
    data = assemble_corpus_localkey_entropy_df(metacorpora_path=metacorpora_path)
    sorted_df = data.sort_values(by=['year'])
    corpus_chronological_order = sorted_df['corpus'].unique().tolist()

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.scatterplot(data=data, x='year', y='entropy', hue='corpus', hue_order=corpus_chronological_order, s=50)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)

    plt.title("Entropy of local keys in each piece")

    if savefig:
        if fig_path is None:
            fig_path = 'monday-inforamtion-theoretic-quantity-figs/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        plt.savefig(fname=fig_path + 'localkey_corpus_entropy-scatter.jpeg', dpi=200, format='jpeg')
    return fig


# 2.  entropy of chord vocab

if __name__ == '__main__':
    result = plot_localkey_entropy_by_corpus(metacorpora_path='dcml_corpora/', fig_path=None)

    # pieceinfo = PieceInfo.from_directory(parent_corpus_path='romantic_piano_corpus/debussy_suite_bergamasque/',
    #                                      piece_name='l075-01_suite_prelude')

    # print(pieceinfo.key_info.local_key.entropy())
    # print(pieceinfo.key_info.local_key.get_changes().entropy())
    # print(pieceinfo.key_info.local_key.unique_labels().entropy())
