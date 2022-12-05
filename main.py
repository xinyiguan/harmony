# Created by Xinyi Guan in 2022.
import os
from typing import Literal, List

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from harmony import util
from harmony.loader import MetaCorporaInfo


def assemble_piece_localkey_entropy_df(metacorpora_path: str,
                                       entropy_type: Literal['full_seq', 'changes_seq', 'unique_seq']) -> pd.DataFrame:
    """
    A dataframe of piece entropy: piecename, entropy, corpus, year, era
    The entropy of key labels in a piece
    """

    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    entropy_df_list = []
    for corpusinfo in metacorpora.meta_info.corpusinfo_list:
        piece_names = pd.Series([item for item in corpusinfo.meta_info.piecename_list], name='piece')
        if entropy_type == 'full_seq':
            piece_localkey_entropy = pd.Series(
                [item.key_info.local_key.entropy() for item in corpusinfo.meta_info.pieceinfo_list],
                name='entropy')

        elif entropy_type == 'changes_seq':
            piece_localkey_entropy = pd.Series(
                [item.key_info.local_key.get_changes().entropy() for item in corpusinfo.meta_info.pieceinfo_list],
                name='entropy')

        elif entropy_type == 'unique_seq':
            piece_localkey_entropy = pd.Series(
                [item.key_info.local_key.unique_labels().entropy() for item in corpusinfo.meta_info.pieceinfo_list],
                name='entropy')
        else:
            raise ValueError(f'Unexpected {entropy_type}')

        piece_year = pd.Series(
            [item.meta_info.composed_end._series.values[0] for item in corpusinfo.meta_info.pieceinfo_list],
            name='year')
        piece_parent_corpus = pd.Series(
            [item.meta_info.corpus_name._series.values[0] for item in corpusinfo.meta_info.pieceinfo_list],
            name='corpus')

        frame = {'piece': piece_names,
                 'entropy': piece_localkey_entropy,
                 'year': piece_year,
                 'corpus': piece_parent_corpus}

        localkey_entropy_df = pd.DataFrame(frame)
        entropy_df_list.append(localkey_entropy_df)
    entropy_df = pd.concat(entropy_df_list)
    entropy_df['era'] = entropy_df['year'].apply(lambda x: util.determine_era_based_on_year(x))
    return entropy_df


def assemble_corpus_localkey_entropy_df(metacorpora_path: str,
                                        entropy_type: Literal['full_seq', 'unique_seq']) -> pd.DataFrame:
    """
    A dataframe of corpus localkey entropy: corpus entropy, year, era
    The entropy of key labels in a corpus.
    """
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    entropy_df_list = []
    for corpusinfo in metacorpora.meta_info.corpusinfo_list:
        if entropy_type == 'full_seq':
            corpus_localkey_entropy = pd.Series(corpusinfo.key_info.local_key.entropy(), name='entropy')
        elif entropy_type == 'unique_seq':
            corpus_localkey_entropy = pd.Series(corpusinfo.key_info.local_key.unique_labels().entropy(), name='entropy')
        else:
            raise ValueError(f'Unexpected {entropy_type}')

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


def plot_attribute_entropy(data: pd.DataFrame, x: str, y: str, hue: str, hue_order: List[str],
                           plot_type: Literal['scatter', 'lmplot'],
                           fig_name: str | None, fig_path: str | None,
                           title: str | None,
                           savefig: bool = True) -> plt.Figure:
    """Assume data has cols: year, corpus, """

    fig, ax = plt.subplots(figsize=(15, 12))
    if plot_type == 'scatter':
        sns.scatterplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)

    elif plot_type == 'lmplot':
        sns.lmplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order)

    else:
        raise NotImplementedError

    plt.title(title)
    plt.tight_layout()
    if savefig:
        if fig_path is None:
            fig_path = 'information-theoretic-quantity-figs/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        plt.savefig(fname=fig_path + fig_name + '.jpeg', dpi=200, format='jpeg')
        print(f'plot saved to {fig_path}{fig_name}.jpeg')
    return fig


if __name__ == '__main__':
    metacorpora_path = 'petit_dcml_corpus/'

    # # __________________________piece_full_seq_entropy_____________________________
    #
    # piece_full_seq_entropy_data = assemble_piece_localkey_entropy_df(metacorpora_path=metacorpora_path,
    #                                                                  entropy_type='full_seq')
    # sorted_df = piece_full_seq_entropy_data.sort_values(by=['year'])
    # corpus_chronological_order = sorted_df['corpus'].unique().tolist()
    #
    # piece_full_seq_entropy_scatter = plot_attribute_entropy(data=piece_full_seq_entropy_data,
    #                                                         x='year', y='entropy', hue='corpus',
    #                                                         hue_order=corpus_chronological_order,
    #                                                         plot_type='scatter',
    #                                                         fig_name='piece_full_seq_entropy_scatter',
    #                                                         fig_path=None,
    #                                                         title='Piece-wise entropy of local key samples',
    #                                                         savefig=True)
    # piece_full_seq_entropy_lmplot = plot_attribute_entropy(data=piece_full_seq_entropy_data,
    #                                                        x='year', y='entropy', hue='corpus',
    #                                                        hue_order=corpus_chronological_order,
    #                                                        plot_type='lmplot',
    #                                                        fig_name='piece_full_seq_entropy_lmplot',
    #                                                        fig_path=None,
    #                                                        title='Piece-wise entropy of local key samples',
    #                                                        savefig=True)
    #
    # # __________________________piece_changes_seq_entropy_____________________________
    #
    # piece_changes_seq_entropy_data = assemble_piece_localkey_entropy_df(metacorpora_path=metacorpora_path,
    #                                                                     entropy_type='changes_seq')
    # sorted_df_2 = piece_changes_seq_entropy_data.sort_values(by=['year'])
    # corpus_chronological_order_2 = sorted_df_2['corpus'].unique().tolist()
    # plot_piece_changes_seq_entropy_scatter = plot_attribute_entropy(data=piece_changes_seq_entropy_data,
    #                                                                 x='year', y='entropy', hue='corpus',
    #                                                                 hue_order=corpus_chronological_order_2,
    #                                                                 plot_type='scatter',
    #                                                                 fig_name='piece_chnages_seq_entropy_scatter',
    #                                                                 fig_path=None,
    #                                                                 title='Piece-wise entropy of key changes',
    #                                                                 savefig=True)
    # plot_piece_changes_seq_entropy_lmplot = plot_attribute_entropy(data=piece_changes_seq_entropy_data,
    #                                                                x='year', y='entropy', hue='corpus',
    #                                                                hue_order=corpus_chronological_order_2,
    #                                                                plot_type='lmplot',
    #                                                                fig_name='piece_changes_seq_entropy_lmplot',
    #                                                                fig_path=None,
    #                                                                title='Piece-wise entropy of unique keys',
    #                                                                savefig=True)
    #
    # # __________________________piece_unique_seq_entropy_____________________________
    #
    # piece_unique_seq_entropy_data = assemble_piece_localkey_entropy_df(metacorpora_path=metacorpora_path,
    #                                                                    entropy_type='unique_seq')
    # sorted_df_3 = piece_unique_seq_entropy_data.sort_values(by=['year'])
    # corpus_chronological_order_3 = sorted_df_3['corpus'].unique().tolist()
    # plot_piece_unique_seq_entropy_scatter = plot_attribute_entropy(data=piece_unique_seq_entropy_data,
    #                                                                x='year', y='entropy', hue='corpus',
    #                                                                hue_order=corpus_chronological_order_3,
    #                                                                plot_type='scatter',
    #                                                                fig_name='piece_unique_seq_entropy_scatter',
    #                                                                fig_path=None,
    #                                                                title='Piece-wise entropy of unique keys',
    #                                                                savefig=True)
    # plot_piece_unique_seq_entropy_lmplot = plot_attribute_entropy(data=piece_unique_seq_entropy_data,
    #                                                               x='year', y='entropy', hue='corpus',
    #                                                               hue_order=corpus_chronological_order_3,
    #                                                               plot_type='lmplot',
    #                                                               fig_name='piece_unique_seq_entropy_lmplot',
    #                                                               fig_path=None,
    #                                                               title='Piece-wise entropy of unique keys',
    #                                                               savefig=True)

    # __________________________corpus_unique_seq_entropy_____________________________
    era_order = ['Renaissance', 'Baroque', 'Classical', 'Romantic']

    corpus_unique_seq_entropy_data = assemble_corpus_localkey_entropy_df(metacorpora_path=metacorpora_path,
                                                                         entropy_type='unique_seq')

    plot_corpus_unique_seq_entropy_scatter = plot_attribute_entropy(data=corpus_unique_seq_entropy_data,
                                                                    x='year', y='entropy', hue='era',
                                                                    hue_order=era_order,
                                                                    plot_type='scatter',
                                                                    fig_name='corpus_unique_seq_entropy_scatter',
                                                                    fig_path=None,
                                                                    title='Corpus-wise entropy of unique keys',
                                                                    savefig=True)
    plot_corpus_unique_seq_entropy_lmplot = plot_attribute_entropy(data=corpus_unique_seq_entropy_data,
                                                                   x='year', y='entropy', hue='era',
                                                                   hue_order=era_order,
                                                                   plot_type='lmplot',
                                                                   fig_name='corpus_unique_seq_entropy_lmplot',
                                                                   fig_path=None,
                                                                   title='Corpus-wise entropy of unique keys',
                                                                   savefig=True)

    # __________________________corpus_full_seq_entropy_____________________________
    corpus_full_seq_entropy_data = assemble_corpus_localkey_entropy_df(metacorpora_path=metacorpora_path,
                                                                       entropy_type='full_seq')

    plot_corpus_full_seq_entropy_scatter = plot_attribute_entropy(data=corpus_full_seq_entropy_data,
                                                                  x='year', y='entropy', hue='era',
                                                                  hue_order=era_order,
                                                                  plot_type='scatter',
                                                                  fig_name='corpus_full_seq_entropy_scatter',
                                                                  fig_path=None,
                                                                  title='Corpus-wise entropy of local keys',
                                                                  savefig=True)

    plot_corpus_full_seq_entropy_lmplot = plot_attribute_entropy(data=corpus_full_seq_entropy_data,
                                                                 x='year', y='entropy', hue='era',
                                                                 hue_order=era_order,
                                                                 plot_type='lmplot',
                                                                 fig_name='corpus_full_seq_entropy_lmplot',
                                                                 fig_path=None,
                                                                 title='Corpus-wise entropy of local keys',
                                                                 savefig=True)
