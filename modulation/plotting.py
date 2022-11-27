# Created by Xinyi Guan in 2022.
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from modulation import util
from modulation.loader import MetaCorpraInfo, CorpusInfo, PieceInfo
from modulation.representation import ModulationBigram


@dataclass
class Modulation:
    metacorpora: MetaCorpraInfo

    # _______________________________________ data _________________________________________

    def _eligible_pieces_list(self, corpus: CorpusInfo) -> List[PieceInfo]:
        eligible_pcs_list = []
        for piece in corpus.annotated_piece_list:
            piece_harmonies_df = piece.get_aspect_df(aspect='harmonies', selected_keys=None)

            # check certain columns exist ('globalkey', 'localkey' columns exists, 'globalkey' column has values):
            if {'globalkey', 'localkey', }.issubset(piece_harmonies_df.columns) and all(
                    piece_harmonies_df['globalkey'].notna()):
                eligible_pcs_list.append(piece.piece_name)
            else:
                pass
        eligible_pieces_list = [PieceInfo(parent_corpus_path=corpus.corpus_path, piece_name=name) for name in
                                eligible_pcs_list]
        return eligible_pieces_list

    def _eligible_corpora_list(self, metacorpora: MetaCorpraInfo) -> List[CorpusInfo]:
        eligible_corpora_list = [CorpusInfo(corpus_path=path) for path in metacorpora.corpus_paths]
        return eligible_corpora_list

    def _corpus_modulation_seq_df(self, corpus: CorpusInfo) -> pd.DataFrame:
        """
        Returns the dataframe with columns: ['corpus', 'fname', 'composed_end', 'localkey_labels', 'num_modulations']
        """
        modulation_df_list = []
        for piece in self._eligible_pieces_list(corpus):
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

    def _piece_modulation_df(self, piece: PieceInfo,
                             fifths: bool = True) -> pd.DataFrame:
        """Get the dataframe containing modulation-relevant [bigram, type, interval, year, era, corpus] of a piece"""

        bigrams_list = piece.get_modulation_bigrams_list()
        # df = []

        modulation_df = pd.DataFrame(data=bigrams_list, columns=['bigram'])
        length = modulation_df.shape[0]
        transition_type = [ModulationBigram.parse(modulation_bigram_str=bigram).type() for bigram in bigrams_list]
        if fifths:
            interval = [ModulationBigram.parse(modulation_bigram_str=bigram).interval().fifths() for bigram in
                        bigrams_list]
        else:
            interval = [ModulationBigram.parse(modulation_bigram_str=bigram).interval() for bigram in bigrams_list]

        modulation_df['type'] = transition_type
        modulation_df['interval'] = interval
        modulation_df['year'] = [piece.composed_year] * length
        modulation_df['era'] = [util.determine_era_based_on_year(piece.composed_year)] * length
        modulation_df['piece'] = [piece.piece_name] * length
        modulation_df['corpus'] = [piece.corpus_name] * length

        if fifths:
            modulation_df = modulation_df.astype({"interval": "int", "year": "int"})
        else:
            modulation_df = modulation_df.astype({"year": "int"})
        return modulation_df

    def _modulation_df(self, data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo],
                       fifths: bool = True) -> pd.DataFrame:
        """
        Transform data into [bigram, type, interval, year, era, corpus] dataframe.
        """

        def _corpus_modulation_df(piece_list) -> pd.DataFrame:
            corpus_modulation_df_list = []
            for piece in piece_list:
                piece_modulation_df = self._piece_modulation_df(piece, fifths=fifths)
                corpus_modulation_df_list.append(piece_modulation_df)
            corpus_modulation_df = pd.concat(corpus_modulation_df_list, ignore_index=True)
            return corpus_modulation_df

        def _metacorpora_modulation_df(corpus_list) -> pd.DataFrame:
            metacorpora_modulation_list = []
            for corpus in corpus_list:
                eligible_piece_list = self._eligible_pieces_list(corpus)
                corpus_modulation_df = _corpus_modulation_df(eligible_piece_list)
                metacorpora_modulation_list.append(corpus_modulation_df)
            metacorpora_modulation_df = pd.concat(metacorpora_modulation_list, ignore_index=True)
            return metacorpora_modulation_df

        if isinstance(data_source, PieceInfo):
            modulation_df = self._piece_modulation_df(data_source, fifths=fifths)
            return modulation_df

        elif isinstance(data_source, CorpusInfo):
            annotated_piece_list = self._eligible_pieces_list(data_source)
            modulation_df = _corpus_modulation_df(annotated_piece_list)
            return modulation_df

        elif isinstance(data_source, MetaCorpraInfo):
            annotated_corpus_list = self._eligible_corpora_list(data_source)
            modulation_df = _metacorpora_modulation_df(annotated_corpus_list)
            return modulation_df

    def _piece_modulation_short_df(self, this_piece: PieceInfo) -> pd.DataFrame:
        """Transform the modulation df into a dataframe with columns: [interval, count, year, era] of a piece"""

        interval_df = self._modulation_df(this_piece)
        modulation_short_df = interval_df['interval'].value_counts().reset_index().rename(
            columns={'index': 'interval', 'interval': 'count'})
        length = modulation_short_df.shape[0]
        modulation_short_df['year'] = [this_piece.composed_year] * length
        modulation_short_df['era'] = [util.determine_era_based_on_year(this_piece.composed_year)] * length
        modulation_short_df['corpus'] = [this_piece.corpus_name] * length
        return modulation_short_df

    def _modulation_short_df(self, data_source: Union[MetaCorpraInfo, CorpusInfo, PieceInfo]) -> pd.DataFrame:
        """Transform data into a dataframe with columns: [interval, count, year, era]"""

        def _corpus_modulation_short_df(this_corpus: CorpusInfo):
            corpus_heatmap_data_list = []
            filtered_piece_list = self._eligible_pieces_list(this_corpus)
            for item in filtered_piece_list:
                piece_heatmap_data = self._piece_modulation_short_df(item)
                corpus_heatmap_data_list.append(piece_heatmap_data)
            corpus_modulation_df = pd.concat(corpus_heatmap_data_list, ignore_index=True)
            return corpus_modulation_df

        def _metacorpora_modulation_short_df(this_metacorpora: MetaCorpraInfo):
            metacorpora_heatmap_data_list = []
            filtered_corpus_list = self._eligible_corpora_list(this_metacorpora)

            for item in filtered_corpus_list:
                corpus_heatmap_data = _corpus_modulation_short_df(item)
                metacorpora_heatmap_data_list.append(corpus_heatmap_data)
            metacorpora_modulation_df = pd.concat(metacorpora_heatmap_data_list, ignore_index=True)
            return metacorpora_modulation_df

        if isinstance(data_source, PieceInfo):
            heatmap_df = self._piece_modulation_short_df(data_source)
            return heatmap_df

        elif isinstance(data_source, CorpusInfo):
            heatmap_df = _corpus_modulation_short_df(data_source)
            return heatmap_df

        elif isinstance(data_source, MetaCorpraInfo):
            heatmap_df = _metacorpora_modulation_short_df(data_source)
            return heatmap_df

    # _______________________________________ plotting _________________________________________
    def plot_chronological_distribution_of_pieces(self, fig_path: Optional[str], savefig: bool = True,
                                                  showfig: bool = False) -> plt.Figure:
        """
        plot the chronological distribution of pieces in the meta-corpus
        """
        composed_years = self.metacorpora.get_corpora_metadata_df(selected_keys=['corpus', 'fnames', 'composed_end'])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(data=composed_years, x='composed_end', stat='count', color='#580F41')

        if savefig:
            if fig_path is None:
                fig_path = '../figs/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            plt.savefig(fname=fig_path + 'pieces_chronological_dist.jpeg', dpi=200, format='jpeg')

        if showfig:
            plt.show()
        return fig

    def plot_modulation_counts_in_a_piece(self, fig_path: Optional[str], savefig: bool = True,
                                          showfig: bool = False) -> sns.relplot():
        """
        plot a replot of modulations count within a piece in the meta-corpus
        """

        # prepare the dataframe for plotting: [corpus, fname, composed_end, localkey_label, num_modulation]
        modulation_df_list = []
        for idx, val in enumerate(self.metacorpora.corpus_name_list):
            corpus = CorpusInfo(corpus_path=self.metacorpora.meta_corpora_path + val + '/')
            corpus_modulation_df = self._corpus_modulation_seq_df(corpus)
            modulation_df_list.append(corpus_modulation_df)

        modulation_df = pd.concat(modulation_df_list, ignore_index=True)

        # for hue order
        corpus_chronological_order = self.metacorpora.get_corpus_list_in_chronological_order()

        # plot the df
        g = sns.relplot(data=modulation_df, x='composed_end', y='num_modulations',
                        height=8, hue='corpus', hue_order=corpus_chronological_order, alpha=0.5)
        if savefig:
            if fig_path is None:
                fig_path = '../figs/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            g.savefig(fname=fig_path + 'ms_count.jpeg', dpi=200, format='jpeg')

        if showfig:
            plt.show()
        return g

    def plot_modulation_interval_distribution(self,
                                              fig_path: Optional[str], savefig: bool = True,
                                              showfig: bool = False) -> sns.JointGrid:
        """
        heatmap/jointplot of modulation steps distribution in the metacorpora.
        """
        data = self._modulation_df(data_source=self.metacorpora, fifths=True)
        min, max = data['interval'].min(), data['interval'].max()
        g = sns.jointplot(data=data, x='year', y='interval', space=0.8,
                          marginal_ticks=True, s=100, marker='s', alpha=0.4, linewidth=0)
        g.ax_joint.set(yticks=list(range(min, max + 1)))
        g.ax_joint.yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

        g.set_axis_labels('Year', 'Interval on the line of fifths')

        g.ax_joint.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

        if savefig:
            if fig_path is None:
                fig_path = '../figs/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            g.savefig(fname=fig_path + 'modulation_interval_distribution.jpeg', dpi=200, format='jpeg')

        if showfig:
            plt.show()

        return g

    def plot_modulation_interval_heatmap(self, fig_path: Optional[str], savefig: bool = True,
                                         showfig: bool = False):

        data = self._modulation_short_df(data_source=self.metacorpora).pivot("interval", "year", "count")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(data, linewidths=.5, ax=ax)

        if savefig:
            if fig_path is None:
                fig_path = '../figs/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            fig.savefig(fname=fig_path + 'ms_heatmap.jpeg', dpi=200, format='jpeg')

        if showfig:
            plt.show()
        return fig

    def plot_modulation_interval_distr_by_modes(self,
                                                fig_path: Optional[str], savefig: bool = True,
                                                showfig: bool = False):
        """
        displot of modulation interval by modes
        """
        data = self._modulation_df(self.metacorpora)
        g = sns.displot(data=data, x='interval', col='type', kind='hist',
                        discrete=True, color='#008080')

        g.set_axis_labels("Interval on the line of fifths", "Count")

        if savefig:
            if fig_path is None:
                fig_path = '../figs/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

        g.savefig(fname=fig_path + 'ms_by_modes.jpeg', dpi=200, format='jpeg')

        if showfig:
            plt.show()

        return g

    def plot_modulation_interval_by_era(self, fig_path: Optional[str], era_order=None,
                                        savefig: bool = True, showfig: bool = False):
        if era_order is None:
            era_order = ['Renaissance', 'Baroque', 'Classical', 'Romantic']

        df = self._modulation_df(data_source=self.metacorpora)

        # Initialize a grid of plots with an Axes for each era
        grid = sns.FacetGrid(df, col="era", height=4,
                             margin_titles=True,
                             col_order=era_order)

        # Draw a bar plot to show the count of modulation steps
        grid.map_dataframe(sns.histplot, data=df, x='interval', stat='count',
                           multiple='stack', hue='corpus', palette='viridis',
                           discrete=True)

        # Set name
        grid.set_axis_labels(x_var='Modulation step (intervals on the line of fifths)', y_var='Count')
        grid.tight_layout()

        if savefig:
            if fig_path is None:
                fig_path = '../figs/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            grid.savefig(fname=fig_path + 'ms_by_era.jpeg', dpi=200, format='jpeg')

        if showfig:
            plt.show()

        return grid

    def plot_modulation_interval_by_modes_by_era(self, fig_path: Optional[str], era_order=None,
                                                 savefig: bool = True, showfig: bool = False):
        if era_order is None:
            era_order = ['Renaissance', 'Baroque', 'Classical', 'Romantic']

        df = self._modulation_df(data_source=self.metacorpora)

        # Initialize a grid of plots with an Axes for each walk
        grid = sns.FacetGrid(df, col="type", row='era', height=4,
                             margin_titles=True,
                             col_order=['MM', 'Mm', 'mM', 'mm'],
                             row_order=era_order)

        # Draw a bar plot to show the count of modulation steps
        grid.map_dataframe(sns.histplot, data=df, x='interval', stat='count',
                           multiple='stack', hue='corpus', palette='viridis',
                           discrete=True)

        # Set name
        grid.set_axis_labels(x_var='Modulation step (intervals on the line of fifths)', y_var='Count')

        # To show the bottom label (interval) in every facet
        for axis in grid.axes.flat:
            axis.tick_params(labelbottom=True)

        grid.tight_layout()

        if savefig:
            if fig_path is None:
                fig_path = '../figs/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            grid.savefig(fname=fig_path + 'ms_by_mode_by_era.jpeg', dpi=200, format='jpeg')

        if showfig:
            plt.show()

        return grid
