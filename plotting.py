from typing import Union

import pandas as pd
from matplotlib import pyplot as plt

from loader import MetaCorpraInfo, CorpusInfo, PieceInfo
import seaborn as sns


def key_modulation_heatmap_data(data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]):
    pass


def transition_prob_heatmap_data():
    pass


if __name__ == '__main__':
    metacorpora_path = 'romantic_piano_corpus/'

    metacorpora = MetaCorpraInfo(metacorpora_path)

    tran_prob = metacorpora.get_transition_matrix(n=2, aspect='harmonies', key='numeral', probability=True)

    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    f, (ax, cbar_ax) = plt.subplots(2, figsize=(20, 20), gridspec_kw=grid_kws)

    sns.heatmap(tran_prob, ax=ax, square=False, vmin=0, vmax=1, cbar=True,
                cbar_ax=cbar_ax, cmap=sns.cubehelix_palette(as_cmap=True), annot=True, fmt='.2f',
                cbar_kws={"orientation": "horizontal", "fraction": 0.1,
                          "label": "Transition probability"})



    ax.set_xlabel("Target chord")
    ax.xaxis.tick_top()
    ax.set_ylabel("Origin chord")
    ax.xaxis.set_label_position('top')
    plt.show()