from typing import Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from loader import MetaCorpraInfo, CorpusInfo, PieceInfo
import modulation
import seaborn as sns


def transition_prob_heatmap_data(transition_prob: pd.DataFrame) -> plt.Figure:
    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    fig, (ax, cbar_ax) = plt.subplots(2, figsize=(20, 20), gridspec_kw=grid_kws)

    sns.heatmap(transition_prob, ax=ax, square=False, vmin=0, vmax=1, cbar=True,
                cbar_ax=cbar_ax, cmap=sns.cubehelix_palette(as_cmap=True), annot=True, fmt='.2f',
                cbar_kws={"orientation": "horizontal", "fraction": 0.1,
                          "label": "Transition probability"})

    ax.set_xlabel("Target chord")
    ax.xaxis.tick_top()
    ax.set_ylabel("Origin chord")
    ax.xaxis.set_label_position('top')
    plt.show()
    return fig


def get_modulation_steps_displot_data(data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]):
    """

    :param data_source:
    :return:
    """

    def get_piecewise_modulation_df(modulations_bigrams_RN):
        MM = modulation.partition_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types='MM')
        MM_steps = modulation.compute_modulation_steps(MM, partition_type='MM', fifths=True)
        MM_modulation_df = pd.DataFrame(MM_steps, columns=['interval'])
        MM_modulation_df['type'] = ['MM'] * MM_modulation_df.shape[0]

        Mm = modulation.partition_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types='Mm')
        Mm_steps = modulation.compute_modulation_steps(Mm, partition_type='Mm', fifths=True)
        Mm_modulation_df = pd.DataFrame(Mm_steps, columns=['interval'])
        Mm_modulation_df['type'] = ['Mm'] * Mm_modulation_df.shape[0]

        mM = modulation.partition_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types='mM')
        mM_steps = modulation.compute_modulation_steps(mM, partition_type='mM', fifths=True)
        mM_modulation_df = pd.DataFrame(mM_steps, columns=['interval'])
        mM_modulation_df['type'] = ['mM'] * mM_modulation_df.shape[0]

        mm = modulation.partition_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types='mm')
        mm_steps = modulation.compute_modulation_steps(mm, partition_type='mm', fifths=True)
        mm_modulation_df = pd.DataFrame(mm_steps, columns=['interval'])
        mm_modulation_df['type'] = ['mm'] * mm_modulation_df.shape[0]

        modulation_df = pd.concat(
            [MM_modulation_df, Mm_modulation_df, mM_modulation_df, mm_modulation_df])
        return modulation_df

    def get_corpuswise_modulation_df(annotated_piece_list):
        corpus_modulation_list = []
        for idx, piece in enumerate(annotated_piece_list):
            piece_modulation_bigrams_RN = piece.get_modulation_bigrams_with_globalkey()
            piece_modulation_df = get_piecewise_modulation_df(piece_modulation_bigrams_RN)
            corpus_modulation_list.append(piece_modulation_df)
        corpus_modulation_df = pd.concat(corpus_modulation_list)
        return corpus_modulation_df

    def get_metacorpora_modulation_df(annotated_corpus_list):
        metacorpora_modulation_list = []
        for idx, corpus in enumerate(annotated_corpus_list):
            corpus_modulation_df = get_corpuswise_modulation_df(corpus.annotated_piece_list)
            metacorpora_modulation_list.append(corpus_modulation_df)
        metacorpora_modulation_df = pd.concat(metacorpora_modulation_list)
        return metacorpora_modulation_df

    if isinstance(data_source, PieceInfo):
        modulations_bigrams_RN = data_source.get_modulation_bigrams_with_globalkey()
        modulation_df = get_piecewise_modulation_df(modulations_bigrams_RN)
        return modulation_df

    elif isinstance(data_source, CorpusInfo):
        annotated_piece_list = data_source.annotated_piece_list
        modulation_df = get_corpuswise_modulation_df(annotated_piece_list)
        return modulation_df


    elif isinstance(data_source, MetaCorpraInfo):
        annotated_corpus_list = data_source.annotated_corpus_list
        modulation_df = get_metacorpora_modulation_df(annotated_corpus_list)
        return modulation_df


if __name__ == '__main__':
    metacorpora_path = 'petit_dcml_corpus/'
    corpus_path = 'romantic_piano_corpus/chopin_mazurkas/'
    # piece = PieceInfo(parent_corpus_path=corpus_path, piece_name='l075-01_suite_prelude')
    # corpus = CorpusInfo(corpus_path)
    metacorpora = MetaCorpraInfo(metacorpora_path)
    result = get_modulation_steps_displot_data(metacorpora)
    print(result)

    sns.displot(data=result, x='interval', col='type', col_wrap=2)

    plt.show()

# https://seaborn.pydata.org/tutorial/categorical.html#categorical-tutorial
