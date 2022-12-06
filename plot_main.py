# Created by Xinyi Guan in 2022.
import os
from typing import Literal, List

import pandas as pd
import pitchtypes
from matplotlib import pyplot as plt
import seaborn as sns
from harmony import util
from harmony.loader import MetaCorporaInfo, SequentialData
from harmony.representation import Key, Numeral, ModulationBigram
from pitchtypes import *


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


def assemble_ms_era_df(metacorpora_path: str) -> pd.DataFrame:
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    # assemble a dataframe (modulation_bigrams_df) with cols: bigram, year, era
    modulation_bigrams_list = []
    for corpusinfo in metacorpora.meta_info.corpusinfo_list:
        corpus_modulation_bigrams_df_list = []
        for pieceinfo in corpusinfo.meta_info.pieceinfo_list:
            if pieceinfo.key_info.local_key.get_changes().len() > 1:
                piece_mod_bigrams = pd.Series(pieceinfo.harmony_info.modulation_bigrams_list(), name='bigram_str',
                                              dtype='object')
                piece_df_len: int = piece_mod_bigrams.shape[0]
                piece_year = pd.Series([pieceinfo.meta_info.composed_end._series[0]] * piece_df_len, name='year',
                                       dtype=int)
                piece_frame = {'bigram_str': piece_mod_bigrams, 'year': piece_year}
                piece_df = pd.DataFrame(piece_frame)
                piece_df['era'] = piece_df['year'].apply(lambda x: util.determine_era_based_on_year(x))
                corpus_modulation_bigrams_df_list.append(piece_df)
            else:
                pass
        if len(corpus_modulation_bigrams_df_list) != 0:
            corpus_df = pd.concat(corpus_modulation_bigrams_df_list)
        else:
            pass
        modulation_bigrams_list.append(corpus_df)
    modulation_bigrams_df = pd.concat(modulation_bigrams_list).sort_values(
        by=['year'], ignore_index=True)  # now bigrams are str, sample row: (g_i_III  1722   Baroque)

    # transform the bigram column from string to ModulationBigram instance and get the interval
    modulation_bigrams_df['interval'] = modulation_bigrams_df['bigram_str'].apply(
        lambda x: ModulationBigram.parse(modulation_bigram_str=x).interval())

    return modulation_bigrams_df


if __name__ == '__main__':
    # metacorpora_path = 'dcml_corpora/'
    # era_order = ['Renaissance', 'Baroque', 'Classical', 'Romantic']

    metacorpora_path = 'dcml_corpora/'

    # __________________________surprisal of modulation step by era_____________________________
    ms_era_df = assemble_ms_era_df(metacorpora_path=metacorpora_path)
    renaissance_df = ms_era_df[ms_era_df['era'] == 'Renaissance'].reset_index(drop=True)
    baroque_df = ms_era_df[ms_era_df['era'] == 'Baroque'].reset_index(drop=True)
    classical_df = ms_era_df[ms_era_df['era'] == 'Classical'].reset_index(drop=True)
    romantic_df = ms_era_df[ms_era_df['era'] == 'Romantic'].reset_index(drop=True)

    renaissance_surprisal_df = SequentialData.from_pd_series(series=renaissance_df['interval']).surprisal().to_frame(
        name='surprisal').reset_index().rename(columns={'index': 'interval'})
    renaissance_surprisal_df['era'] = ['Renaissance'] * renaissance_surprisal_df.shape[0]

    baroque_surprisal_df = SequentialData.from_pd_series(series=baroque_df['interval']).surprisal().to_frame(
        name='surprisal').reset_index().rename(columns={'index': 'interval'})
    baroque_surprisal_df['era'] = ['Baroque'] * baroque_surprisal_df.shape[0]

    classical_surprisal_df = SequentialData.from_pd_series(series=classical_df['interval']).surprisal().to_frame(
        name='surprisal').reset_index().rename(columns={'index': 'interval'})
    classical_surprisal_df['era'] = ['Classical'] * classical_surprisal_df.shape[0]

    romantic_surprisal_df = SequentialData.from_pd_series(series=romantic_df['interval']).surprisal().to_frame(
        name='surprisal').reset_index().rename(columns={'index': 'interval'})
    romantic_surprisal_df['era'] = ['Romantic'] * romantic_surprisal_df.shape[0]

    surprisal_df = pd.concat([renaissance_surprisal_df, baroque_surprisal_df, classical_surprisal_df,
                              romantic_surprisal_df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = util.set_plot_style_palette_4()
    surprisal_df['interval(fifth)'] = surprisal_df['interval'].apply(
        lambda x: pitchtypes.SpelledIntervalClass(str(x)).fifths())

    sns.pointplot(ax=ax, x='interval(fifth)', y='surprisal', data=surprisal_df, hue='era', palette=palette)

    # sns.lineplot(data=renaissance_surprisal_series, color="#046586")
    # sns.lineplot(data=baroque_surprisal_series, color="#28A9A1")
    # sns.lineplot(data=classical_surprisal_series, color="#F4A016")
    # sns.lineplot(data=romantic_surprisal_series, color="#F6BBC6")
    #
    plt.title('Surprisal (information content) of modulation steps')
    plt.tight_layout()
    plt.savefig(fname='information-theoretic-quantity-figs/' + 'era-ms-surprisal.jpeg', dpi=200,
                format='jpeg')

    # # __________________________corpus_full_seq_entropy: violinplot_____________________________
    # corpus_full_seq_entropy_data = assemble_corpus_localkey_entropy_df(metacorpora_path=metacorpora_path,
    #                                                              entropy_type='full_seq')
    # corpus_full_seq_entropy_data = corpus_full_seq_entropy_data.sort_values(by=['year'])
    # palette = util.set_plot_style_palette_4()
    # plt.figure(figsize=(12, 6))
    # plt.title('Violin plot of entropy distribution of corpus-wise local key (full samples)')
    # sns.violinplot(data=corpus_full_seq_entropy_data, x="era", y="entropy",order=era_order, palette=palette)
    # plt.savefig(fname='information-theoretic-quantity-figs/' + 'corpus_full_seq_entropy_violinplot.jpeg', dpi=200, format='jpeg')

    # # __________________________piece_full_seq_entropy: violinplot_____________________________
    #
    # piece_full_seq_entropy_data = assemble_piece_localkey_entropy_df(metacorpora_path=metacorpora_path,
    #                                                                  entropy_type='full_seq')
    # palette = util.set_plot_style_palette_4()
    # plt.figure(figsize=(12, 6))
    # plt.title('Violin plot of entropy distribution of piece-wise local key (full samples)')
    # sns.violinplot(data=piece_full_seq_entropy_data, x="era", y="entropy",order=era_order, palette=palette)
    # plt.savefig(fname='information-theoretic-quantity-figs/' + 'piece_full_seq_entropy_violinplot.jpeg', dpi=200, format='jpeg')

    # # __________________________piece_unique_seq_entropy_____________________________
    # piece_unique_seq_entropy_data = assemble_piece_localkey_entropy_df(metacorpora_path=metacorpora_path,
    #                                                                    entropy_type='unique_seq')
    # piece_unique_seq_entropy_data = piece_unique_seq_entropy_data.sort_values(by=['year'])
    # corpus_chronological_order = piece_unique_seq_entropy_data['corpus'].unique().tolist()

    # # scatter plot:
    # fig, ax = plt.subplots(figsize=(15, 12))
    # sns.scatterplot(data=piece_unique_seq_entropy_data, x='year', y='entropy', hue='corpus',
    #                 hue_order=corpus_chronological_order, s=40, alpha=0.7)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)
    # plt.title('Piece-wise entropy of unique keys')
    # plt.tight_layout()
    # plt.savefig(fname='information-theoretic-quantity-figs/' + 'piece_unique_seq_entropy_scatter.jpeg', dpi=200,
    #             format='jpeg')

    # # violin plot:
    # palette = util.set_plot_style_palette_4()
    # plt.figure(figsize=(10, 7))
    # plt.title('Violin plot of entropy distribution of piece-wise unique local key labels')
    # sns.violinplot(data=piece_unique_seq_entropy_data, x="era", y="entropy", order=era_order, palette=palette)
    # plt.tight_layout()
    # plt.savefig(fname='information-theoretic-quantity-figs/' + 'piece_unique_seq_entropy_violinplot.jpeg', dpi=200,
    #             format='jpeg')

    # # __________________________corpus_unique_seq_entropy_____________________________
    #
    # corpus_unique_seq_entropy_data = assemble_corpus_localkey_entropy_df(metacorpora_path=metacorpora_path,
    #                                                                      entropy_type='unique_seq')
    #
    # corpus_unique_seq_entropy_data = corpus_unique_seq_entropy_data.sort_values(by=['year'])
    # corpus_chronological_order = corpus_unique_seq_entropy_data['corpus'].unique().tolist()

    # # scatter plot:
    # fig, ax = plt.subplots(figsize=(15, 12))
    # sns.scatterplot(data=corpus_unique_seq_entropy_data, x='year', y='entropy', hue='corpus',
    #                 hue_order=corpus_chronological_order, s=100, alpha=0.7)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)
    # plt.title('Corpus-wise entropy of unique keys')
    # plt.tight_layout()
    # plt.savefig(fname='information-theoretic-quantity-figs/' + 'corpus_unique_seq_entropy_scatter.jpeg', dpi=200,
    #             format='jpeg')

    # # violin plot:
    # palette = util.set_plot_style_palette_4()
    # plt.figure(figsize=(10,7))
    # plt.title('Violin plot of entropy distribution of corpus-wise unique local key labels')
    # sns.violinplot(data=corpus_unique_seq_entropy_data, x="era", y="entropy", order=era_order, palette=palette)
    # plt.tight_layout()
    # plt.savefig(fname='information-theoretic-quantity-figs/' + 'corpus_unique_seq_entropy_violinplot.jpeg', dpi=200,
    #             format='jpeg')
