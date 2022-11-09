#  By Xinyi Guan, 2022.
# This script is to create relevant figs for the modulation_notes.md file

from loader import PieceInfo, CorpusInfo, MetaCorpraInfo
import plotting, util
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Define the meta-corpus to work with
metacorpora_path = 'petit_dcml_corpus/'
metacorpora = MetaCorpraInfo(metacorpora_path)


#  1. plot the chronological distribution of pieces in the meta-corpus
def plot_chronological_distribution_of_pieces() -> plt.Figure:
    composed_years = metacorpora.get_corpora_concat_metadata_df(selected_keys=['corpus', 'fnames', 'composed_end'])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(data=composed_years, x='composed_end', stat='count', color='#580F41', kde=True)
    plt.savefig(fname='figs/pieces_chronological_dist.jpeg', dpi=200, format='jpeg')
    return fig


#  2. plot modulations count within a piece in the meta-corpus
def plot_modulation_counts_in_a_piece() -> sns.JointGrid:
    # prepare the dataframe for plotting: [corpus, fname, composed_end, localkey_label, num_modulation]
    modulation_df_list = []
    for idx, val in enumerate(metacorpora.corpus_name_list):
        corpus = CorpusInfo(corpus_path=metacorpora_path + val + '/')
        corpus_modulation_df = corpus.get_modulation_data_df()
        modulation_df_list.append(corpus_modulation_df)

    modulation_data_df = pd.concat(modulation_df_list, ignore_index=True)

    # plot the df
    sns.set_theme(style="ticks")
    g = sns.JointGrid(data=modulation_data_df, x='composed_end', y='num_modulations',
                      height=8, hue='corpus', marginal_ticks=True, )
    g.plot_joint(sns.scatterplot, s=30, alpha=0.6)
    g.plot_marginals(sns.histplot)
    g.savefig(fname='figs/modulation_count_in_pieces.jpeg', dpi=200, format='jpeg')

    return g


#  3. plot Distribution of modulations

## 3.1 jointplot of modulation steps distribution in the metacorpus
def jointplot_modulation_steps_distribution_in_metacorpus() -> sns.JointGrid:
    data = plotting.get_modulation_steps_data(data_source=metacorpora)
    g = sns.jointplot(data=data, x='year', y='interval', kind='hist', bins=30)
    g.set_axis_labels('Year', 'Interval on the line of fifths')
    g.savefig(fname='figs/jointplot_modulation_steps_distribution_in_metacorpus.jpeg', dpi=200, format='jpeg')
    return g


## 3.2. View modulation steps dist in 4 types of mode transitions
def displot_modulation_steps_dist_4_modes():
    data = plotting.get_modulation_steps_with_transition_types_data(metacorpora)
    g = sns.displot(data=data, x='interval', col='type', kind='hist',
                    kde=True, discrete=True, color='#008080')

    g.set_axis_labels("Interval on the line of fifths", "Count")
    g.savefig(fname='figs/displot_modulation_steps_dist_4_modes.jpeg', dpi=200, format='jpeg')
    return g


## 3.3 facetgrid_modulation_steps_by_era:
def facetgrid_modulation_steps_by_era():
    df = plotting.get_modulation_steps_with_transition_types_data(data_source=metacorpora)

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="era", height=4,
                         margin_titles=True,
                         col_order=['Renaissance', 'Baroque', 'Classical', 'Early Rom.', 'Late Rom.'])

    # Draw a bar plot to show the count of modulation steps
    grid.map_dataframe(sns.histplot, data=df, x='interval', stat='count',
                       multiple='stack', hue='corpus', palette='YlGnBu',
                       discrete=True)

    # Set name
    grid.set_axis_labels(x_var='Modulation step (intervals on the line of fifths)', y_var='Count')

    grid.tight_layout()

    grid.savefig(fname='figs/facetgrid_modulation_steps_by_era.jpeg', dpi=200, format='jpeg')

    return grid


## 3.4. facetgrid of modulation steps with modes and era data
def facetgrid_modulation_steps_with_modes_by_era():
    df = plotting.get_modulation_steps_with_transition_types_data(data_source=metacorpora)

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="type", row='era', height=4,
                         margin_titles=True,
                         col_order=['MM', 'Mm', 'mM', 'mm'],
                         row_order=['Renaissance', 'Baroque', 'Classical', 'Early Rom.', 'Late Rom.'])

    # Draw a bar plot to show the count of modulation steps
    grid.map_dataframe(sns.histplot, data=df, x='interval', stat='count',
                       multiple='stack', hue='corpus', palette='YlGnBu',
                       discrete=True)

    # Set name
    grid.set_axis_labels(x_var='Modulation step (intervals on the line of fifths)', y_var='Count')

    # To show the bottom label (interval) in every facet
    for axis in grid.axes.flat:
        axis.tick_params(labelbottom=True)

    grid.tight_layout()

    grid.savefig(fname='figs/facetgrid_modulation_steps_with_modes_by_era.jpeg', dpi=200, format='jpeg')

    return grid


if __name__ == '__main__':
    plot_chronological_distribution_of_pieces()
    plot_modulation_counts_in_a_piece()
    jointplot_modulation_steps_distribution_in_metacorpus()
    displot_modulation_steps_dist_4_modes()
    facetgrid_modulation_steps_by_era()
    facetgrid_modulation_steps_with_modes_by_era()

# TODO: NEED TO FILTER THE PIECES FIRST! MUST HAVE GLOBALKEY INFO
