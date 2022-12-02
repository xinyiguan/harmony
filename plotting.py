from dataclasses import dataclass
from typing import Union, List
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import util
from harmony.loader import MetaCorpraInfo, CorpusInfo, PieceInfo


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

    # elif 1817 < year < 1857:
    #     return 'Early Rom.'
    #
    elif 1817 < year < 1931:
        return 'Romantic'


@dataclass
class ModulationSteps:
    metacorpora: MetaCorpraInfo

    @staticmethod
    def get_eligible_pieces_list(corpus: CorpusInfo):
        # corpus_harmonies_df = corpus.harmonies_df

        eligible_pieces_list = []
        for piece in corpus.annotated_piece_list:
            piece_harmonies_df = piece.get_aspect_df(aspect='harmonies', selected_keys=None)

            # check certain columns exist ('globalkey', 'localkey' columns exists, 'globalkey' column has values):
            if {'globalkey', 'localkey', }.issubset(piece_harmonies_df.columns) and all(
                    piece_harmonies_df['globalkey'].notna()):
                eligible_pieces_list.append(piece.piece_name)
            else:
                pass
        eligible_pieces_list = [PieceInfo(parent_corpus_path=corpus.corpus_path, piece_name=name) for name in
                                eligible_pieces_list]
        return eligible_pieces_list

    @staticmethod
    def get_eligible_corpora_list(metacorpora: MetaCorpraInfo) -> List[CorpusInfo]:
        eligible_corpora_list = [CorpusInfo(corpus_path=path) for path in metacorpora.corpus_paths]
        return eligible_corpora_list

    def get_corpuswise_modulation_data_df(self, corpus: CorpusInfo) -> pd.DataFrame:
        modulation_df_list = []
        for piece in self.get_eligible_pieces_list(corpus):
            corpus_name = piece.corpus_name
            fname = piece.piece_name
            composed_end_yr = \
                corpus.metadata_df[corpus.metadata_df['fnames'] == piece.piece_name]['composed_end'].values[0]
            localkey_labels = '-'.join(piece.get_localkey_label_list())
            num_modulations = len(piece.get_localkey_label_list())
            modulation_data_list = list([corpus_name, fname, composed_end_yr, localkey_labels, num_modulations])
            modulation_df_list.append(modulation_data_list)

        modulation_df = pd.DataFrame(modulation_df_list,
                                     columns=['corpus', 'fname', 'composed_end', 'localkey_labels', 'num_modulations'])
        return modulation_df

    def piecewise_modulation_data_with_transition_types_df(self, piece: PieceInfo, modulations_bigrams_RN: List,
                                                           steps_in_fifths: bool = True):
        """Get the dataframe containing harmony-relevant [bigram, interval, type, year, era, corpus] of a piece"""
        dfs = []
        for modulation_type in ['MM', 'Mm', 'mM', 'mm']:
            bigrams = util.filter_modulation_bigrams_by_types(modulation_bigrams_list=modulations_bigrams_RN,
                                                              modulation_type=modulation_type)
            modulation_df = pd.DataFrame(bigrams, columns=['bigram'])
            steps = util.compute_modulation_steps(bigrams, fifths=steps_in_fifths)
            modulation_df['interval'] = steps
            length = modulation_df.shape[0]
            modulation_df['type'] = [modulation_type] * length
            modulation_df['year'] = [piece.composed_year] * length
            modulation_df['era'] = [determine_era_based_on_year(piece.composed_year)] * length
            modulation_df['corpus'] = [piece.corpus_name] * length
            dfs.append(modulation_df)

        all_modulation_df = pd.concat(dfs)
        all_modulation_df = all_modulation_df.astype({"interval": "int", "year": "int"})
        return all_modulation_df

    def get_modulation_steps_with_transition_types_data(self,
                                                        data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]):
        """Transform data into [bigram, interval, type, year, era, corpus] dataframe. Dataframe['interval] = ['m3, M3, p4]"""

        def get_corpuswise_modulation_df(annotated_piece_list):
            corpus_modulation_list = []
            for piece in annotated_piece_list:
                piece_modulation_bigrams_RN = piece.get_modulation_bigrams_with_globalkey()
                piece_modulation_df = self.piecewise_modulation_data_with_transition_types_df(piece,
                                                                                              piece_modulation_bigrams_RN)
                corpus_modulation_list.append(piece_modulation_df)
            corpus_modulation_df = pd.concat(corpus_modulation_list)
            return corpus_modulation_df

        def get_metacorpora_modulation_df(annotated_corpus_list):
            metacorpora_modulation_list = []
            for corpus in annotated_corpus_list:
                # get eligible pieces:
                annotated_piece_list = self.get_eligible_pieces_list(corpus)
                corpus_modulation_df = get_corpuswise_modulation_df(annotated_piece_list)
                metacorpora_modulation_list.append(corpus_modulation_df)
            metacorpora_modulation_df = pd.concat(metacorpora_modulation_list)
            return metacorpora_modulation_df

        if isinstance(data_source, PieceInfo):
            modulations_bigrams_RN = data_source.get_modulation_bigrams_with_globalkey()
            modulation_df = self.piecewise_modulation_data_with_transition_types_df(data_source, modulations_bigrams_RN)
            return modulation_df

        elif isinstance(data_source, CorpusInfo):
            annotated_piece_list = self.get_eligible_pieces_list(data_source)
            modulation_df = get_corpuswise_modulation_df(annotated_piece_list)
            return modulation_df

        elif isinstance(data_source, MetaCorpraInfo):
            annotated_corpus_list = self.get_eligible_corpora_list(data_source)
            modulation_df = get_metacorpora_modulation_df(annotated_corpus_list)
            return modulation_df

    def piecewise_modulation_steps_data_df(self, this_piece: PieceInfo):
        interval_df = self.get_modulation_steps_with_transition_types_data(this_piece)
        modulation_data_df = interval_df['interval'].value_counts().reset_index().rename(
            columns={'index': 'interval', 'interval': 'count'})
        length = modulation_data_df.shape[0]
        modulation_data_df['year'] = [this_piece.composed_year] * length
        modulation_data_df['era'] = [determine_era_based_on_year(this_piece.composed_year)] * length
        modulation_data_df['corpus'] = [this_piece.corpus_name] * length
        return modulation_data_df

    def get_modulation_steps_data(self, data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]):
        """Transform data into a dataframe with columns: [interval, count, year, era]"""

        def get_corpuswise_modulation_steps_data(this_corpus: CorpusInfo):
            corpus_heatmap_data_list = []
            filtered_piece_list = self.get_eligible_pieces_list(this_corpus)
            # filtered_pieceinfo_list = [PieceInfo(parent_corpus_path=this_corpus.corpus_path, piece_name=name)
            #                                           for name in filtered_piece_list]
            for item in filtered_piece_list:
                piece_heatmap_data = self.piecewise_modulation_steps_data_df(item)
                corpus_heatmap_data_list.append(piece_heatmap_data)
            corpus_modulation_df = pd.concat(corpus_heatmap_data_list)
            return corpus_modulation_df

        def get_metacorpora_modulation_steps_data(this_metacorpora: MetaCorpraInfo):
            metacorpora_heatmap_data_list = []
            filtered_corpus_list = self.get_eligible_corpora_list(this_metacorpora)
            # filtered_corpusinfo_list= [CorpusInfo(corpus_path=path) for path in [this_metacorpora.meta_corpora_path + val + '/' for val in filtered_corpus_list]]

            for item in filtered_corpus_list:
                corpus_heatmap_data = get_corpuswise_modulation_steps_data(item)
                metacorpora_heatmap_data_list.append(corpus_heatmap_data)
            metacorpora_heatmap_data = pd.concat(metacorpora_heatmap_data_list)
            return metacorpora_heatmap_data

        if isinstance(data_source, PieceInfo):
            heatmap_df = self.piecewise_modulation_steps_data_df(data_source)
            return heatmap_df

        elif isinstance(data_source, CorpusInfo):
            heatmap_data = get_corpuswise_modulation_steps_data(data_source)
            return heatmap_data

        elif isinstance(data_source, MetaCorpraInfo):
            heatmap_data = get_metacorpora_modulation_steps_data(data_source)
            return heatmap_data

    def plot_chronological_modulation_steps_facetgrid(self):
        df = self.get_modulation_steps_with_transition_types_data(data_source=self.metacorpora)

        # Initialize a grid of plots with an Axes for each walk
        grid = sns.FacetGrid(df, col="type", row='era', height=4,
                             margin_titles=True,
                             col_order=['MM', 'Mm', 'mM', 'mm'],
                             row_order=['Baroque', 'Classical', 'Early Rom.', 'Late Rom.'])

        # Draw a bar plot to show the count of harmony steps
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
    metacorpora_path = 'dcml_corpora/'

    metacorpora = MetaCorpraInfo(metacorpora_path)

    m = ModulationSteps(metacorpora)

    modulation_df_list = []
    for idx, val in enumerate(metacorpora.corpus_name_list):
        corpus = CorpusInfo(corpus_path=metacorpora_path+ val + '/')
        corpus_modulation_df = m.get_corpuswise_modulation_data_df(corpus)
        modulation_df_list.append(corpus_modulation_df)
    df = pd.concat(modulation_df_list, ignore_index=True)

    filtered = df.loc[(df['num_modulations'] > 25)]

    sorted = df.sort_values(by=['composed_end'])



    result = print(filtered.to_markdown())

    print(result)
