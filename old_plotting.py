import os
from typing import Literal

import pandas as pd
from matplotlib import colors

import loader
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    im_ratio = data.shape[0] / data.shape[1]
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_aspect('auto')

    return im, cbar


def heatmap_plot_key_corpora_stats(metacorpora_path: str, key: str, stat_type: str,
                                   index_sorting_method: Literal['count', 'fifth'],
                                   fig_title: str ='Heatmap of chord vocab probabilities across current corpora',
                                   lognorm: bool = True):
    """
    plot the heatmap of key (such as chord vocab or 'numeral') stats (probs or information content) across corpus in the metacorpora
    :param metacorpora_path:
    :param key:
    :param stat_type: 'probs', 'ic'
    :return:
    """

    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 15

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    metacorpora = loader.MetaCorporaInfo(metacorpora_path=metacorpora_path)
    # subcorpus_list = metacorpora.subcorpus_list
    # unique_key_vals = metacorpora.get_unique_values_by_key(key=key)

    sorted_vertical_index_list = metacorpora.get_sorted_key_value_list(key=key, method=index_sorting_method)
    sorted_corpus_list = metacorpora.get_sorted_subcorpus_by_year(output_format='corpus')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    fig.suptitle(fig_title)

    if stat_type == 'probs':
        prob_df_list = []
        for idx, subcorpus_name in enumerate(sorted_corpus_list):
            subcorpus_path = metacorpora_path + subcorpus_name + '/'
            subcorpus = metacorpora.get_subcorpus(subcorpus_path)
            keyvalues_probs = subcorpus.compute_probs(key=key)
            extended_keyvalues_probs = pd.DataFrame(data=keyvalues_probs,
                                                    index=sorted_vertical_index_list)  # extend the prob table, standardize

            current_col_name = extended_keyvalues_probs.columns.values.tolist()[0]
            extended_keyvalues_probs = extended_keyvalues_probs.rename(
                columns={current_col_name: subcorpus_name})  # rename the column

            prob_df_list.append(extended_keyvalues_probs)
        concat_prob_df = pd.concat(prob_df_list, axis=1)
        concat_prob_df = concat_prob_df.fillna(0)  # fill nan with 0
        current_index_list = concat_prob_df.index.values

        vertical_index_permutation = []
        for idx, val in enumerate(sorted_vertical_index_list):
            pos = \
                [i for i in range(len(current_index_list)) if current_index_list[i] == sorted_vertical_index_list[idx]][
                    0]
            vertical_index_permutation.append(pos)

        prob_nparray = concat_prob_df.to_numpy()
        print(type(prob_nparray))
        sorted_prob_nparray = prob_nparray[vertical_index_permutation]

        if lognorm:
            im, cbar = heatmap(sorted_prob_nparray, row_labels=sorted_vertical_index_list,
                               col_labels=sorted_corpus_list, ax=ax,
                               norm=colors.LogNorm(), cmap="PuBu", cbarlabel="probabilities")
        else:
            im, cbar = heatmap(sorted_prob_nparray, row_labels=sorted_vertical_index_list,
                               col_labels=sorted_corpus_list, ax=ax,
                               cmap="PuBu", cbarlabel="probabilities")

        # colorbar scale: norm=colors.LogNorm()
        # fig.tight_layout()
        plt.show()


    elif stat_type == 'ic':
        raise NotImplementedError
    else:
        raise NotImplementedError


## sns ===========================================================================

def sns_heatmap_key_corpora_stats(metacorpora_path: str, key: str, stat_type: str,
                                  index_sorting_method: Literal['count', 'fifth'], fig_title:str):
    raise NotImplementedError


def sns_lineplot_key_entropy_in_metacorpora(metacorpora_path: str, key: str, fig_title:str):
    metacorpora = loader.MetaCorporaInfo(metacorpora_path=metacorpora_path)
    subcorpus_list = metacorpora.subcorpus_list

    sorted_key_index_list = metacorpora.get_sorted_key_value_list(key=key, method='count')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    fig.suptitle(fig_title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    chord_vocab_entropy_list = []
    corpus_name_list = []
    for idx, val in enumerate(metacorpora.subcorpus_paths):
        corpus = loader.CorpusInfo(corpus_path=val)
        corpus_name = val.split(os.sep)[-2]
        chord_vocab_entropy = corpus.compute_entropy(key=key)
        chord_vocab_entropy_list.append(chord_vocab_entropy)
        corpus_name_list.append(corpus_name)

    chord_vocab_entropy_array = np.array(chord_vocab_entropy_list)

    sorted_corpus_list = metacorpora.get_sorted_subcorpus_by_year(output_format='corpus')

    horizontal_index_permutation = []
    for idx, val in enumerate(sorted_corpus_list):
        pos = [i for i in range(len(corpus_name_list)) if corpus_name_list[i] == sorted_corpus_list[idx]][0]
        horizontal_index_permutation.append(pos)

    sorted_entropy_array = chord_vocab_entropy_array[horizontal_index_permutation]

    data = pd.DataFrame(sorted_entropy_array, sorted_corpus_list, columns=['entropies'])
    print(data)

    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.show()




if __name__ == '__main__':
    metacorpora_path = 'romantic_piano_corpus/'
    metacorpora = loader.MetaCorporaInfo(metacorpora_path)
    heatmap_plot_key_corpora_stats(metacorpora_path, key='numeral', stat_type='probs', index_sorting_method='count',)


    # chopin_corpus = loader.CorpusInfo(corpus_path)
    # harmonies_df = chopin_corpus.get_corpus_harmonies_df(attach_metadata=True)

    # heatmap_plot_key_corpora_stats(metacorpora_path='romantic_piano_corpus/', key='numeral', stat_type='probs', index_sorting_method='count')
