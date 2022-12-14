import collections
import typing
from dataclasses import dataclass
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from musana.generics import Sequential
from musana.loader import PieceInfo, MetaCorporaInfo
from musana.musictypes import TonalHarmony


@dataclass
class ChordTransitionAnalysis:
    """
    This class holds the analysis and plots for studying the chord transitions (bigrams).
    """
    metacorpora: MetaCorporaInfo

    # define typings:
    NGram = typing.Tuple[TonalHarmony]
    HarmonyCondition = typing.Callable[[TonalHarmony, ], bool]
    NGramCondition = typing.Callable[[NGram, ], bool]

    # define misc:
    cmap_dict = {'M': 'rocket_r', 'm': 'mako_r'}

    def plot_heatmaps_to_folder(self, folder_path: str) -> None:
        eras = ['Renaissance', 'Baroque', 'Classical', 'Romantic']
        dpi = 200

        self.plot_metacorpora_chord_transitions_heatmap(n=2, view_top_n=30).savefig(
            fname=f'{folder_path}metacorpora_chord_transitions',
            dpi=dpi)

        self.plot_chord_transition_in_by_era_heatmap(n=2, eras=eras, view_top_n=30).savefig(
            fname=f'{folder_path}chord_transitions_by_era',
            dpi=dpi)

    def plot_metacorpora_chord_transitions_heatmap(self, n: int, view_top_n: int) -> plt.Figure:
        modes = ['M', 'm']
        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(6 * 2, 5 * 2 * 2))
        fig.suptitle('Chord transitions in metacorpora')

        for mode, ax in zip(modes, axs):
            cmap = self.cmap_dict[mode]
            pieceinfo_list = metacorpora.get_annotated_pieces()
            transition_matrix = self.transition_matrix(pieceinfo_list=pieceinfo_list, n=n, mode=mode)

            ax.set_title(f'{mode}, shape={transition_matrix.shape}')
            self.transition_matrix_heatmap(transition_matrix=transition_matrix, ax=ax, cmap=cmap, view_top_n=view_top_n)
        return fig

    def plot_chord_transition_in_by_era_heatmap(self, n: int, eras: List[str], view_top_n: int) -> plt.Figure:
        axs_keys = [[f'{era}-{mode}' for era in eras] for mode in ['M', 'm']]  # create internal names for axes.

        fig, axs = plt.subplot_mosaic(axs_keys, figsize=(6 * 2 * 4, 5 * 2 * 2))
        fig.suptitle('Chord transition by era')

        for key, ax in axs.items():
            era, mode = key.split('-')
            cmap = self.cmap_dict[mode]
            era_condition = lambda piece_info: piece_info.meta_info.era == era
            pieceinfo_list = metacorpora.filter_pieces_by_condition(condition=era_condition)
            transition_matrix = self.transition_matrix(pieceinfo_list=pieceinfo_list, n=n, mode=mode)

            ax.set_title(f'{era}, shape={transition_matrix.shape}')
            self.transition_matrix_heatmap(transition_matrix=transition_matrix, ax=ax, cmap=cmap, view_top_n=view_top_n)
        return fig

    def transition_matrix_heatmap(self, transition_matrix: pd.DataFrame, ax: plt.Axes, view_top_n: int,
                                  cmap: str = 'rocket_r') -> None:

        if transition_matrix.size > 0:
            # view the top n rank chord symbol
            transition_matrix_to_view = transition_matrix.iloc[:view_top_n, :view_top_n]

            sns.heatmap(data=100 * transition_matrix_to_view, ax=ax, xticklabels=True, yticklabels=True, cmap=cmap,
                        annot=True, fmt='.1f', annot_kws={'fontsize': 'xx-small'})
            ax.xaxis.tick_top()
            ax.tick_params(axis='x', rotation=45)

    def transition_matrix(self, pieceinfo_list: List[PieceInfo], n: int,
                          mode: typing.Literal['M', 'm'], probability: bool = True) -> pd.DataFrame:
        """
        Get the transition matrix as pd.Dataframe
        """

        tonal_harmony_transitions = self.get_tonal_harmony_transition_in_mode_segment(
            pieceinfo_list=pieceinfo_list, mode=mode, n=n)

        chord_str_transitions = tonal_harmony_transitions.map(
            operation=lambda _tuple: '-'.join([x.chord_str for x in _tuple]))

        sources, targets = zip(*[(ngram.rsplit('-', 1)[0], ngram.rsplit('-', 1)[1])
                                 for ngram in chord_str_transitions])

        count_sources = collections.Counter(sources)

        if n > 2:
            count_targets = collections.Counter(targets)
        elif n == 2:
            count_targets = collections.Counter(sources)
        else:
            raise ValueError(f'n must be greater than 1. Unexpected value {n=}')

        transition_matrix = pd.DataFrame(0, columns=count_targets.keys(), index=count_sources.keys())

        for source, target in zip(sources, targets):
            transition_matrix.loc[source, target] += 1

        if probability:
            transition_prob = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
            return transition_prob
        return transition_matrix

    def get_tonal_harmony_transition_in_mode_segment(self, pieceinfo_list: List[PieceInfo],
                                                     mode: typing.Literal['M', 'm'],
                                                     n: int = 2) -> Sequential:
        """This function will return the Sequential of tuples of """
        harmony_condition = lambda tonal_harmony: tonal_harmony.chord_str != ''
        same_local_key = lambda _tuple: all((x.localkey == _tuple[0].localkey for x in _tuple))
        in_same_mode = lambda _tuple: all((x.localkey.quality == mode for x in _tuple))

        n_gram_condition = lambda x: all((f(x) for f in [
            same_local_key,
            in_same_mode
        ]))
        tonal_harmony_transitions = self.piececwise_tonal_harmony_ngrams_from_pieceinfo_list(
            pieceinfo_list=pieceinfo_list,
            harmony_condition=harmony_condition,
            n_gram_condition=n_gram_condition,
            n=n)
        return tonal_harmony_transitions

    @staticmethod
    def piececwise_tonal_harmony_ngrams_from_pieceinfo_list(pieceinfo_list: List[PieceInfo],
                                                            harmony_condition: HarmonyCondition,
                                                            n_gram_condition: NGramCondition, n: int) -> Sequential:
        """
        Perform piece-wise operation (get TonalHarmony ngrams) from the pieceinfo_list according to
        pre-defined rules and join the ngrams as a Sequential object.
        """

        # define piece-wise operation   PieceInfo -> Sequential
        def piece_operation(piece_info: PieceInfo) -> Sequential:
            return piece_info.get_tonal_harmony_sequential \
                .filter_by_condition(condition=harmony_condition) \
                .get_n_grams(n=n) \
                .filter_by_condition(condition=n_gram_condition)

        sequentials_list = [piece_operation(pieceinfo) for pieceinfo in pieceinfo_list]
        chord_transitions = Sequential.join(sequentials=sequentials_list)
        return chord_transitions

    # ________________________________________________________________________________________________
    def symmetry_measure(self,a, b, transition_matrix: pd.DataFrame):
        # For a tuple of TonalHarmony object, symmetry of a chord transition is defined as:
        # sym_mode(a, b) = min_mode{p(a->b)/p(b->a), p(b->a)/p(a->b)}

        return min(transition_matrix[a][b] / transition_matrix[b][a], transition_matrix[b][a] / transition_matrix[a][b])

    def compute_metacorpora_symmetry_measures(self):

        transition_matrix = ...  # your transition matrix

        for a in range(len(transition_matrix)):
            for b in range(len(transition_matrix)):
                sym = self.symmetry_measure(a, b, transition_matrix)
                print(f"The symmetry measure for the transition from state {a} to state {b} is {sym}.")


class ChordAnalysis:
    pass


if __name__ == '__main__':
    metacorpora_path = 'naive_corpora/'
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    chord_transition_analysis = ChordTransitionAnalysis(metacorpora=metacorpora)

    chord_transition_analysis.plot_heatmaps_to_folder(folder_path='chord_transitions_heatmap/')
