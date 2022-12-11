import collections
import typing
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from harmony.generics import Sequential
from harmony.loader import PieceInfo, MetaCorporaInfo
from harmony.musictypes import TonalHarmony
import seaborn as sns


@dataclass
class ChordTransitionAnalysis:
    metacorpora: MetaCorporaInfo

    NGram = typing.Tuple[TonalHarmony]
    HarmonyCondition = typing.Callable[[TonalHarmony, ], bool]
    NGramCondition = typing.Callable[[NGram, ], bool]

    def plot_to_folder(self, folder_path: str) -> None:
        eras = ['Renaissance', 'Baroque', 'Classical', 'Romantic']
        dpi = 200
        # self.plot_chord_transition_in_major_segment().savefig(fname=f'{folder_path}chord_transition_in_major_segment',
        #                                                       dpi=dpi)
        # self.plot_chord_transition_in_minor_segment().savefig(fname=f'{folder_path}chord_transition_in_minor_segment',
        #                                                       dpi=dpi)
        # self.plot_chord_transition_in_major_segment_by_era(eras=eras).savefig(
        #     fname=f'{folder_path}chord_transition_in_major_segment_by_era', dpi=dpi)
        # self.plot_chord_transition_in_minor_segment_by_era(eras=eras).savefig(
        #     fname=f'{folder_path}chord_transition_in_minor_segment_by_era', dpi=dpi)

        self.plot_metacorpora_chord_transitions().savefig(fname=f'{folder_path}metacorpora_chord_transitions', dpi=dpi)
        self.plot_chord_transition_in_by_era(eras=eras).savefig(fname=f'{folder_path}chord_transitions_by_era', dpi=dpi)

    def plot_metacorpora_chord_transitions(self) -> plt.Figure:
        modes = ['M', 'm']
        cmap_dict = {'M': 'rocket_r', 'm': 'mako_r'}
        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(6 * 2, 5 * 2 * 2))
        fig.suptitle('Chord transitions in metacorpora')

        for mode, ax in zip(modes, axs):
            cmap = cmap_dict[mode]
            tonal_harmony_transitions = self.get_tonal_harmony_transition_in_segment(
                pieceinfos=metacorpora.get_annotated_pieces(),
                mode=mode,
                n=1)
            chord_str_transitions = tonal_harmony_transitions.map(
                operation=lambda _tuple: '-'.join([x.chord_str for x in _tuple]))
            transition_matrix = chord_str_transitions.get_transition_matrix(probability=True)
            print(transition_matrix.shape)
            ax.set_title(f'{mode}, shape={transition_matrix.shape}')
            self.plot_transition_matrix(transition_matrix=transition_matrix, ax=ax, cmap=cmap)
        return fig

    def plot_chord_transition_in_by_era(self, eras: List[str]) -> plt.Figure:
        axs_keys = [[f'{era}-{mode}' for era in eras] for mode in ['M', 'm']]
        cmap_dict = {'M': 'rocket_r', 'm': 'mako_r'}

        fig, axs = plt.subplot_mosaic(axs_keys, figsize=(6 * 2 * 4, 5 * 2 * 2))
        fig.suptitle('Chord transition by era')

        for key, ax in axs.items():
            era, mode = key.split('-')
            cmap = cmap_dict[mode]
            era_condition = lambda piece_info: piece_info.meta_info.era == era
            piece_infos = metacorpora.filter_pieces_by_condition(condition=era_condition)
            tonal_harmony_transitions = self.get_tonal_harmony_transition_in_segment(
                pieceinfos=piece_infos,
                mode=mode,
                n=1)
            chord_str_transitions = tonal_harmony_transitions.map(
                operation=lambda _tuple: '-'.join([x.chord_str for x in _tuple]))
            transition_matrix = chord_str_transitions.get_transition_matrix(probability=True)
            ax.set_title(f'{era}, shape={transition_matrix.shape}')
            self.plot_transition_matrix(transition_matrix=transition_matrix, ax=ax, cmap=cmap)
        return fig

    def plot_chord_distribution(self) -> plt.Figure:
        piece_infos = self.metacorpora.get_annotated_pieces()
        tonal_harmonies = self.tonal_harmonies_from_pieceinfos(pieceinfos=piece_infos)
        chord_strs = tonal_harmonies.map(operation=str)._seq
        count = collections.Counter(chord_strs).most_common()
        count = np.array(count)
        counts = np.array(count[:, 1], dtype=int)
        probabilities = counts / counts.sum()
        series = pd.Series(probabilities, index=count[:, 0])
        fig, ax = plt.subplots(figsize=(6, 5))
        self.plot_series(series=series, ax=ax)
        return fig

    def plot_transition_matrix(self, transition_matrix: pd.DataFrame, ax: plt.Axes, top_n: int = 30,
                               cmap: str = 'rocket_r') -> None:

        if transition_matrix.size > 0:
            # view the top n rank chord symbol
            transition_matrix_to_view = transition_matrix.iloc[:top_n, :top_n]

            # for cmap: 'rocket_r' or 'mako_r'
            sns.heatmap(data=100 * transition_matrix_to_view, ax=ax, xticklabels=True, yticklabels=True, cmap=cmap,
                        annot=True, fmt='.1f', annot_kws={'fontsize': 'xx-small'})
            ax.xaxis.tick_top()
            ax.tick_params(axis='x', rotation=45)

    def plot_series(self, series: pd.Series, ax: plt.Axes, top_n: int = 30, cmap: str = 'rocket_r'):
        if series.size > 0:
            df_from_series = series.to_frame()

            df_to_view = df_from_series.iloc[:top_n, [0]]

            sns.lineplot(data=df_to_view, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            # sns.heatmap(data=100 * df_to_view, ax=ax, yticklabels=True, cmap=cmap,
            #            annot=True, fmt='.1f', annot_kws={'fontsize': 'xx-small'},cbar=False)

    def get_tonal_harmony_transition_in_segment(self, pieceinfos: List[PieceInfo], mode: typing.Literal['M', 'm'],
                                                n: int = 2) -> Sequential:
        harmony_condition = lambda tonal_harmony: tonal_harmony.chord_str != ''
        same_local_key = lambda _tuple: all((x.localkey == _tuple[0].localkey for x in _tuple))
        in_major_mode = lambda _tuple: all((x.localkey.quality == mode for x in _tuple))

        n_gram_condition = lambda x: all((f(x) for f in [
            same_local_key,
            in_major_mode
        ]))
        tonal_harmony_transitions = self.tonal_harmony_transitions_from_pieceinfos(pieceinfos=pieceinfos,
                                                                                   harmony_condition=harmony_condition,
                                                                                   n_gram_condition=n_gram_condition,
                                                                                   n=n)
        return tonal_harmony_transitions

    @staticmethod
    def tonal_harmony_transitions_from_pieceinfos(pieceinfos: List[PieceInfo], harmony_condition: HarmonyCondition,
                                                  n_gram_condition: NGramCondition, n: int = 2) -> Sequential:
        # define piece-wise operation   PieceInfo -> Sequential
        def piece_operation(piece_info: PieceInfo) -> Sequential:
            sequential = piece_info.get_tonal_harmony_sequential \
                .filter_by_condition(condition=harmony_condition) \
                .get_n_grams(n=n) \
                .filter_by_condition(condition=n_gram_condition)
            return sequential

        chord_transitions = Sequential.join(sequentials=[piece_operation(pieceinfo) for pieceinfo in pieceinfos])
        return chord_transitions

    @staticmethod
    def tonal_harmonies_from_pieceinfos(pieceinfos: List[PieceInfo],
                                        harmony_condition: HarmonyCondition = lambda x: True) -> Sequential:
        def piece_operation(piece_info: PieceInfo) -> Sequential:
            sequential = piece_info.get_tonal_harmony_sequential \
                .filter_by_condition(condition=harmony_condition)
            return sequential

        harmonies = Sequential.join(sequentials=[piece_operation(pieceinfo) for pieceinfo in pieceinfos])
        return harmonies


if __name__ == '__main__':
    metacorpora_path = 'petit_dcml_corpus/'
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    chord_transition_analysis = ChordTransitionAnalysis(metacorpora=metacorpora)

    # chord_transition_analysis.plot_metacorpora_chord_transitions()

    chord_transition_analysis.plot_to_folder(folder_path='chord_transition_heatmaps/')

    # fig = chord_transition_analysis.plot_chord_distribution()
    # plt.show(dpi=200)
