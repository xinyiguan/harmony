from typing import Union, List
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import util
from loader import MetaCorpraInfo, CorpusInfo, PieceInfo


def transition_prob_heatmap(transition_prob: pd.DataFrame) -> plt.Figure:
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


# ERA_ORDER = {'Renaissance': 0, 'Baroque': 1, 'Classical': 2, 'Early Rom.': 3, 'Late Rom.': 4}
ERA_ORDER = ['Renaissance', 'Baroque', 'Classical', 'Early Rom.', 'Late Rom.']


def determine_era_based_on_year(year) -> str:
    if 0 < year < 1650:
        return 'Renaissance'

    elif 1649 < year < 1759:
        return 'Baroque'

    elif 1758 < year < 1819:
        return 'Classical'

    elif 1817 < year < 1857:
        return 'Early Rom.'

    elif 1856 < year < 1931:
        return 'Late Rom.'


def piecewise_modulation_data_with_transition_types_df(piece: PieceInfo, modulations_bigrams_RN: List):
    """Get the dataframe containing modulation-relevant [interval, type, year, era, corpus] of a piece"""
    dfs = []
    for partition_type in ['MM', 'Mm', 'mM', 'mm']:
        bigrams = util.filter_modualtion_bigrams_by_types(modulations_bigrams_RN, partition_types=partition_type)
        steps = util.compute_modulation_steps(bigrams, partition_type=partition_type, fifths=True)
        modulation_df = pd.DataFrame(steps, columns=['interval'])
        length = modulation_df.shape[0]
        modulation_df['type'] = [partition_type] * length
        modulation_df['year'] = [piece.composed_year] * length
        modulation_df['era'] = [determine_era_based_on_year(piece.composed_year)] * length
        modulation_df['corpus'] = [piece.corpus_name] * length
        dfs.append(modulation_df)

    all_modulation_df = pd.concat(dfs)
    return all_modulation_df


def get_modulation_steps_with_transition_types_data(data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]):
    """Transform data into [interval, type, year, era, corpus] dataframe. Dataframe['interval] = ['m3, M3, p4]"""

    def get_corpuswise_modulation_df(annotated_piece_list):
        corpus_modulation_list = []
        for idx, piece in enumerate(annotated_piece_list):
            piece_modulation_bigrams_RN = piece.get_modulation_bigrams_with_globalkey()
            piece_modulation_df = piecewise_modulation_data_with_transition_types_df(piece, piece_modulation_bigrams_RN)
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
        modulation_df = piecewise_modulation_data_with_transition_types_df(data_source, modulations_bigrams_RN)
        return modulation_df

    elif isinstance(data_source, CorpusInfo):
        annotated_piece_list = data_source.annotated_piece_list
        modulation_df = get_corpuswise_modulation_df(annotated_piece_list)
        return modulation_df

    elif isinstance(data_source, MetaCorpraInfo):
        annotated_corpus_list = data_source.annotated_corpus_list
        modulation_df = get_metacorpora_modulation_df(annotated_corpus_list)
        return modulation_df


def piecewise_modulation_steps_data_df(this_piece: PieceInfo):
    interval_df = get_modulation_steps_with_transition_types_data(this_piece)
    modulation_data_df = interval_df['interval'].value_counts().reset_index().rename(
        columns={'index': 'interval', 'interval': 'count'})
    length = modulation_data_df.shape[0]
    modulation_data_df['year'] = [this_piece.composed_year] * length
    modulation_data_df['era'] = [determine_era_based_on_year(this_piece.composed_year)] * length
    modulation_data_df['corpus'] = [this_piece.corpus_name] * length
    return modulation_data_df


def get_modulation_steps_data(data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]):
    """Transform data into a dataframe with columns: [interval, count, year, era]"""

    def get_corpuswise_modulation_steps_data(annotated_piece_list):
        corpus_heatmap_data_list = []
        for idx, piece in enumerate(annotated_piece_list):
            piece_heatmap_data = piecewise_modulation_steps_data_df(piece)
            corpus_heatmap_data_list.append(piece_heatmap_data)
        corpus_modulation_df = pd.concat(corpus_heatmap_data_list)
        return corpus_modulation_df

    def get_metacorpora_modulation_steps_data(annotated_corpus_list):
        metacorpora_heatmap_data_list = []
        for idx, corpus in enumerate(annotated_corpus_list):
            corpus_heatmap_data = get_corpuswise_modulation_steps_data(corpus.annotated_piece_list)
            metacorpora_heatmap_data_list.append(corpus_heatmap_data)
        metacorpora_heatmap_data = pd.concat(metacorpora_heatmap_data_list)
        return metacorpora_heatmap_data

    if isinstance(data_source, PieceInfo):
        heatmap_df = piecewise_modulation_steps_data_df(data_source)
        return heatmap_df

    elif isinstance(data_source, CorpusInfo):
        annotated_piece_list = data_source.annotated_piece_list
        heatmap_data = get_corpuswise_modulation_steps_data(annotated_piece_list)
        return heatmap_data

    elif isinstance(data_source, MetaCorpraInfo):
        annotated_corpus_list = data_source.annotated_corpus_list
        heatmap_data = get_metacorpora_modulation_steps_data(annotated_corpus_list)
        return heatmap_data


def plot_chronological_modulation_steps_facetgrid(data_source: MetaCorpraInfo):
    df = get_modulation_steps_with_transition_types_data(data_source=data_source)

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="type", row='era', height=4,
                         margin_titles=True,
                         col_order=['MM', 'Mm', 'mM', 'mm'],
                         row_order=['Baroque', 'Classical', 'Early Rom.', 'Late Rom.'])

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

    return grid


if __name__ == '__main__':
    metacorpora_path = 'petit_dcml_corpus/'
    corpus_path = 'romantic_piano_corpus/debussy_suite_bergamasque/'
    # piece = PieceInfo(parent_corpus_path=corpus_path, piece_name='l075-01_suite_prelude')
    # corpus = CorpusInfo(corpus_path)
    metacorpora = MetaCorpraInfo(metacorpora_path)
    # result = get_modulation_steps_displot_data(metacorpora)

    # result = get_modulation_steps_with_transition_types_data(metacorpora)
    # print(result)

    plot_chronological_modulation_steps_facetgrid(metacorpora)
