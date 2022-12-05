#  By Xinyi Guan on 03 December 2022.
import os
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from harmony import util
from harmony.loader import MetaCorporaInfo, PieceInfo


def assemble_piece_localkey_entropy_df(metacorpora_path: str):
    """A dataframe of piece entropy: piecename, entropy, composer, year, era"""

    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    entropy_df_list = []
    for corpusinfo in metacorpora.meta_info.corpusinfo_list:
        piece_names = pd.Series([item for item in corpusinfo.meta_info.piecename_list], name='piece')
        piece_localkey_entropy = pd.Series(
            [item.key_info.local_key.entropy() for item in corpusinfo.meta_info.pieceinfo_list], name='entropy')
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

    fig, ax = plt.subplots(figsize=(20, 12))
    sns.scatterplot(data=data, x='year', y='entropy', hue='corpus', hue_order=corpus_chronological_order)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.title("Entropy of local keys in each piece")

    if savefig:
        if fig_path is None:
            fig_path = '../inforamtion-theoretic-quantity-figs/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        plt.savefig(fname=fig_path + 'localkey_piece_entropy.jpeg', dpi=200, format='jpeg')
    return fig



def assemble_corpus_localkey_entropy_df(metacorpora_path: str):
    pass