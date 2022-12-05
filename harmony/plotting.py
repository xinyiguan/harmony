#  By Xinyi Guan on 03 December 2022.
import os
from typing import List, Literal

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


def plot_attribute_entropy(data: pd.DataFrame, x: str, y: str, hue: str, hue_order: str,
                           plot_type: Literal['scatter'],
                           fig_name: str | None, fig_path: str | None,
                           title: str | None,
                           savefig: bool = True) -> plt.Figure:
    """Assume data has cols: year, corpus, """

    fig, ax = plt.subplots(figsize=(16, 12))
    if plot_type == 'scatter':
        sns.scatterplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    else:
        raise NotImplementedError

    plt.title(title)
    if savefig:
        if fig_path is None:
            fig_path = '../information-theoretic-quantity-figs/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        plt.savefig(fname=fig_path + fig_name + '.jpeg', dpi=200, format='jpeg')
        print(f'plot saved to {fig_path}{fig_name}.jpeg')
    return fig
