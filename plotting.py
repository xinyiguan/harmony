from typing import Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from loader import MetaCorpraInfo, CorpusInfo, PieceInfo
import seaborn as sns


def key_modulation_heatmap_data(data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]):
    pass


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


if __name__ == '__main__':
    metacorpora_path = 'petit_dcml_corpus/'
    metacorpora = MetaCorpraInfo(metacorpora_path)

    modulation_df_list = []
    for idx, val in enumerate(metacorpora.corpus_name_list):
        corpus = CorpusInfo(corpus_path=metacorpora_path + val + '/')
        corpus_modulation_df = corpus.get_modulation_data_df()
        modulation_df_list.append(corpus_modulation_df)

    modulation_df = pd.concat(modulation_df_list, ignore_index=True)
    print(modulation_df)
