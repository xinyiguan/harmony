from typing import List

import dimcat
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import math as math
import compute


def data_df_for_plots(metacorpora_path: str, num_of_corpus: int = None) -> pd.DataFrame:
    list_of_corpus = [dimcat.data.Corpus(directory=path) for path in
                      compute.get_subcorpus_path(metacorpora_path=metacorpora_path)[:num_of_corpus]]
    corpus_names_list = [corpus.data.keys()[0] for corpus in list_of_corpus]

    probs = [compute.compute_probs(corpus=corpus, key='numeral') for corpus in list_of_corpus]
    ic = [compute.compute_information_content(corpus=corpus, key='numeral') for corpus in list_of_corpus]
    entropies = [compute.compute_entropy(corpus=corpus, key='numeral') for corpus in list_of_corpus]
    composition_yr = [compute.compute_corpus_mean_composition_year(corpus=corpus) for corpus in list_of_corpus]

    plot_data_list = []
    for idx, val in enumerate(probs):
        data_df = probs[idx].to_frame()  # add ic column
        data_df['ic'] = ic[idx]
        data_df['year'] = composition_yr[idx]
        n_events = data_df.shape[0]
        corpus_name = [corpus_names_list[idx]] * n_events
        data_df['corpus'] = corpus_name
        data_df = data_df.reset_index()
        mapping = {data_df.columns[0]: 'numeral', data_df.columns[1]: 'probability', data_df.columns[2]: 'ic',
                   data_df.columns[3]: 'year', data_df.columns[4]: 'corpus'}
        data_df = data_df.rename(columns=mapping)
        plot_data_list.append(data_df)

    concat_data_df = pd.concat(plot_data_list)
    print(concat_data_df)

    return concat_data_df


def plot_heatmap(metacorpora_path: str) -> plt.Figure:
    concat_data_df = data_df_for_plots(metacorpora_path)

    heatmap_df = concat_data_df[['year', 'numeral', 'probability']]
    heatmap_df=heatmap_df.pivot(index='numeral', columns='year', values='probability').fillna(0)
   # print(heatmap_df)
    log_norm = LogNorm(vmin=heatmap_df['probability'].min().min(), vmax=heatmap_df['probability'].max().max())
    cbar_ticks = [math.pow(10, i) for i in
                  range(math.floor(math.log10(heatmap_df['probability'].min().min())), 1 + math.ceil(math.log10(heatmap_df['probability'].max().max())))]

    fig, ax = plt.subplots()
    ax = sns.heatmap(data=heatmap_df, norm=log_norm, cbar_kws={"ticks": cbar_ticks})
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    metacorpora_path = 'romantic_piano_corpus/'

    # concat_data_df = data_df_for_plots(metacorpora_path, num_of_corpus=2)
    # fig, ax = plt.subplots()
    # ax = sns.jointplot(data=concat_data_df, x='year', y='probability', hue='numeral', kind='hist')
    # fig.tight_layout()
    # plot_heatmap(metacorpora_path)
    fig=plot_heatmap(metacorpora_path)
    fig.show()


    # my_corpus_path = 'romantic_piano_corpus/chopin_mazurkas/'
    # my_corpus = dimcat.data.Corpus(directory=my_corpus_path)
    # print(my_corpus.data)
