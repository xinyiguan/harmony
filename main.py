import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dimcat

import compute
import dimcat_plot



def task1():
    metacorpora_path = 'romantic_piano_corpus/'
    n = 5
    list_of_corpus = [dimcat.data.Corpus(directory=path) for path in
                      compute.get_subcorpus_path(metacorpora_path=metacorpora_path)[:n]]
    corpus_names = [corpus.data.keys()[0] for corpus in list_of_corpus]
    entropies = [compute.compute_information_content(corpus=corpus, key='numeral') for corpus in list_of_corpus]

    # add a column of corpus name to the df
    heatmap_data_list=[]
    for idx, val in enumerate(entropies):
        heatmap_df=entropies[idx].to_frame().reset_index()
        n_events = heatmap_df.shape[0]
        corpus_name = [corpus_names[idx]]*n_events
        heatmap_df['corpus']=corpus_name
        mapping = {heatmap_df.columns[0]: 'numeral', heatmap_df.columns[1]: 'entropy', heatmap_df.columns[2]:'corpus'}
        heatmap_df=heatmap_df.rename(columns=mapping)
        heatmap_data_list.append(heatmap_df)

    concat_heatmap_df = pd.concat(heatmap_data_list)
    concat_heatmap_df=concat_heatmap_df.pivot(index='numeral', columns='corpus', values='entropy')
    print(concat_heatmap_df)

    fig, ax = plt.subplots()
    ax = sns.heatmap(concat_heatmap_df)
    fig.tight_layout()
    plt.show()

    # # define a mapping to sort the numerals for the plot to look more "continuous"..
    # sorted_numeral_list=entropies[0].index.values    # based on the order in the first corpus
    # item_index=[]
    # for idx, val in enumerate(sorted_numeral_list):
    #     item_index = np.where(array == val)

    # fig = plotting.plot_entropy_across_corpus(corpus_names=corpus_names,entropies=entropies)
    # fig.show()


if __name__ == '__main__':
    task1()