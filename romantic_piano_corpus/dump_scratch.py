from typing import Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from loader import MetaCorpraInfo, CorpusInfo, PieceInfo
import modulation
import seaborn as sns

def get_modulation_steps_facetgrid_data(data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]):
    """

    :param data_source:
    :return:
    """

    if isinstance(data_source, PieceInfo):
        modulations_bigrams_RN = data_source.get_modulation_bigrams_with_globalkey()

        MM = modulation.partition_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types='MM')
        MM_steps = modulation.compute_modulation_steps(MM, partition_type='MM')
        MM_modulation_df = pd.DataFrame(MM_steps, columns=['interval'])
        MM_modulation_df['type'] = ['MM']*MM_modulation_df.shape[0]

        Mm = modulation.partition_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types='Mm')
        Mm_steps = modulation.compute_modulation_steps(Mm, partition_type='Mm')
        Mm_modulation_df = pd.DataFrame(Mm_steps, columns=['interval'])
        Mm_modulation_df['type'] = ['Mm']*Mm_modulation_df.shape[0]

        mM = modulation.partition_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types='mM')
        mM_steps = modulation.compute_modulation_steps(mM, partition_type='mM')
        mM_modulation_df = pd.DataFrame(mM_steps, columns=['interval'])
        mM_modulation_df['type'] = ['mM']*mM_modulation_df.shape[0]

        mm = modulation.partition_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types='mm')
        mm_steps = modulation.compute_modulation_steps(mm, partition_type='mm')
        mm_modulation_df = pd.DataFrame(mm_steps, columns=['interval'])
        mm_modulation_df['type'] = ['mm']*mm_modulation_df.shape[0]

        modulation_df = pd.concat(
            [MM_modulation_df, Mm_modulation_df, mM_modulation_df, mm_modulation_df])

        return modulation_df


    elif isinstance(data_source, CorpusInfo):
        raise NotImplementedError
    elif isinstance(data_source, MetaCorpraInfo):
        raise NotImplementedError


def key_modulation_steps_facetgrid(modulation_data: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    sns.histplot(ax=axes[0, 0], data=modulation_data, x='interval')
    axes[0, 0].set_title('MM')
    sns.histplot(ax=axes[0, 1], data=modulation_data.loc['Mm'])
    axes[0, 1].set_title('Mm')
    sns.histplot(ax=axes[1, 0], data=modulation_data.loc['mM'])
    axes[1, 0].set_title('mM')
    sns.histplot(ax=axes[1, 1], data=modulation_data.loc['mm'])
    axes[1, 1].set_title('mm')

    plt.show()
