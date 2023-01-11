import collections
import typing
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import musana.generics as generics
from musana.loader import PieceInfo, MetaCorporaInfo
from musana.harmony_types import TonalHarmony
import musana.plotting as plotting


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
            transition_matrix = self.chord_str_transition_matrix(pieceinfo_list=pieceinfo_list, n=n, mode=mode,
                                                                 probability=True)

            ax.set_title(f'{mode}, shape={transition_matrix.shape}')
            plotting.transition_matrix_heatmap(transition_matrix=transition_matrix, ax=ax, cmap=cmap,
                                               view_top_n=view_top_n)
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
            transition_matrix = self.chord_str_transition_matrix(pieceinfo_list=pieceinfo_list, n=n, mode=mode,
                                                                 probability=True)

            ax.set_title(f'{era}, shape={transition_matrix.shape}')
            plotting.transition_matrix_heatmap(transition_matrix=transition_matrix, ax=ax, cmap=cmap,
                                               view_top_n=view_top_n)
        return fig

    def chord_str_transition_matrix(self, chord_str_transitions: generics.ST, probability: bool = True) -> pd.DataFrame:
        # Create a TransitionMatrix instance with the n-grams
        transition_matrix = generics.TransitionMatrix(chord_str_transitions).create_matrix(probability=probability)
        return transition_matrix

    def chord_str_transitions_sequential(self, pieceinfo_list: List[PieceInfo], n: int,
                                         mode: typing.Literal['M', 'm']) -> generics.ST:
        # Get the tonal harmony transitions in the given mode segment
        tonal_harmony_transitions = self.get_tonal_harmony_transitions(pieceinfo_list, mode, n)

        # For bigrams, add additional filter on tonal_harmony_transition such that a != b
        if n == 2:
            filtered_tonal_harmony_transitions = generics.Sequential.from_sequence(
                sequence=[tonal_harmony for tonal_harmony in tonal_harmony_transitions if
                          tonal_harmony[0].chord_str != tonal_harmony[1].chord_str])
            tonal_harmony_transitions = filtered_tonal_harmony_transitions

        # Extract the chord str from the tonal harmony tuple as a sequential of tuples,
        chord_str_transitions = tonal_harmony_transitions.map(
            operation=lambda _tuple: tuple([x.chord_str for x in _tuple]))

        return chord_str_transitions

    def get_tonal_harmony_transitions(self, pieceinfo_list: List[PieceInfo],
                                      mode: typing.Literal['M', 'm'],
                                      n: int = 2) -> generics.Sequential:
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
                                                            n_gram_condition: NGramCondition,
                                                            n: int) -> generics.Sequential:
        """
        Perform piece-wise operation (get TonalHarmony ngrams) from the pieceinfo_list according to
        pre-defined rules and join the ngrams as a Sequential object.
        """

        # define piece-wise operation   PieceInfo -> Sequential
        def piece_operation(piece_info: PieceInfo) -> generics.Sequential:
            return piece_info.get_tonal_harmony_sequential \
                .filter_by_condition(condition=harmony_condition) \
                .get_n_grams(n=n) \
                .filter_by_condition(condition=n_gram_condition)

        sequentials_list = [piece_operation(pieceinfo) for pieceinfo in pieceinfo_list]
        chord_transitions = generics.Sequential.join(sequentials=sequentials_list)
        return chord_transitions


@dataclass
class ChordBigramsAnalysis:
    """
    This class holds the symmetry measure of bigrams given a Sequential of bigram tuples.
    """
    bigram_tuples_sequential: generics.ST

    # def symmetry_measures_matrix(self, transition_matrix: pd.DataFrame) -> pd.DataFrame:
    #     mask = transition_matrix + transition_matrix.T == 0  # mask all (a,b) s.t. T(a,b)=T(b,a)=0, T:= transition_matrix
    #     T = np.ma.array(data=transition_matrix, mask=mask)
    #     T[T == 0] = 1e-20  # Set all 0 probabilities with epsilon
    #     E = T / T.T  # T(a,b)/T(b,a) = p(a->b)/p(b->a)
    #     S = np.min(np.ma.stack([E, 1 / E]),
    #                axis=0)  # min {p(a->b)/p(b->a),p(b->a)/p(a->b)} = min(E, 1 / E), element-wise min
    #
    #     df = pd.DataFrame(data=S, index=transition_matrix.index, columns=transition_matrix.columns)
    #     return df

    def relative_frequency(self, top_n: typing.Optional[int]):
        """plot the relative frequency of bigrams, rank in the x-axis and proportion in y-axis
        relative frequency = occurrences of the bigram/total bigrams occurrences count
        """
        bigram_counter = collections.Counter(self.bigram_tuples_sequential).most_common()
        xs = 1 + np.arange(len(bigram_counter))
        total_count = sum([tuple[1] for tuple in bigram_counter])
        ys = [tuple[1] / total_count for tuple in bigram_counter]

        plt.scatter(x=xs[:top_n], y=ys[:top_n], c='#1f77b4', alpha=0.5)
        plt.title('Relative frequency of bigrams')
        plt.xlabel("Rank")
        plt.ylabel("Proportion")
        plt.show()

    def symmetry_measures_matrix_conditional_prob_version(self) -> pd.DataFrame:
        # For a tuple of TonalHarmony object, symmetry of a chord transition is defined as:
        # sym_mode(a, b) = min_mode{p(a|b)/p(b|a), p(b|a)/p(a|b)} (from the transition matrix)
        transition_matrix = generics.TransitionMatrix(n_grams=self.bigram_tuples_sequential).create_matrix()
        mask = transition_matrix + transition_matrix.T == 0  # mask all (a,b) s.t. T(a,b)=T(b,a)=0, T:= transition_matrix
        ma = np.ma.array(data=transition_matrix, mask=mask)
        ma = np.ma.where(ma == 0, 1e-20, ma)  # Set all 0 probabilities with epsilon
        E = ma / ma.T  # T(a,b)/T(b,a) = p(a->b)/p(b->a)
        S = np.min(np.ma.stack([E, 1 / E]),
                   axis=0)  # min {p(a->b)/p(b->a),p(b->a)/p(a->b)} = min(E, 1 / E), element-wise min

        symmetry_measures_matrix = pd.DataFrame(data=S, index=transition_matrix.index,
                                                columns=transition_matrix.columns)
        return symmetry_measures_matrix

    def mode_symmetry_measure_conditional_prob_version(self):
        """
        The mode symmetry is defined as the average over all bigram symmetries of chord progressions in segments in this mode.
        sym(m) = sum(p_m(a->b)*sym_m(a, b))
        """
        df = self.symmetry_measures_matrix_conditional_prob_version()

        T = generics.TransitionMatrix(n_grams=self.bigram_tuples_sequential).create_matrix()
        M = np.tril(T + T.T)
        S = np.ma.array(data=df, mask=df.isna())
        mode_sym = np.sum(S * np.tril(M))

        return mode_sym

    def fabian_symmetry_measures_matrix(self):
        """
        sym_mode(a, b) = min_mode{p(a->b)/p(b->a), p(b->a)/p(a->b)} where p(a->b)=count(a->b)/N_m
        sym(a, b)=min{p(ab)/p(ba), p(ba)/p(ab)} => min{count(a->b)/count(b->a), count(b->a)/count(a->b)}
        """
        count_transition_matrix = generics.TransitionMatrix(n_grams=self.bigram_tuples_sequential).create_matrix(
            probability=False)
        mask = count_transition_matrix + count_transition_matrix.T == 0  # mask all (a,b) s.t. T(a,b)=T(b,a)=0, T:= transition_matrix
        ma = np.ma.array(data=count_transition_matrix, mask=mask)
        ma = np.ma.where(ma == 0, 1e-20, ma)  # Set all 0 probabilities with epsilon
        E = ma / ma.T  # T(a,b)/T(b,a) = p(a->b)/p(b->a)
        S = np.min(np.ma.stack([E, 1 / E]),
                   axis=0)  # min {p(a->b)/p(b->a),p(b->a)/p(a->b)} = min(E, 1 / E), element-wise min

        symmetry_measures_matrix = pd.DataFrame(data=S, index=count_transition_matrix.index,
                                                columns=count_transition_matrix.columns)

        return symmetry_measures_matrix

    def fabian_mode_symmetry_measures(self):
        count_transition_matrix = generics.TransitionMatrix(n_grams=self.bigram_tuples_sequential).create_matrix(
            probability=False)
        symmetry_measures_matrix = self.fabian_symmetry_measures_matrix()

        # entries in the matrix: countm(a ! b)/Nm, where Nm is the number of segments in that mode
        N_m = len(self.bigram_tuples_sequential)
        normalized_transition_matrix: pd.DataFrame = count_transition_matrix / N_m

        element_wise_matrix_mult = normalized_transition_matrix.mul(other=symmetry_measures_matrix,
                                                                    fill_value=0)  # p(a->b) * sym(a, b)

        mode_sym = np.sum(element_wise_matrix_mult.to_numpy())
        return mode_sym

    def phik_corr(self):
        raise NotImplementedError

    # new measures: ------
    def directionality(self) -> float:
        count_ab = ...
        count_ba = ...

        z: complex = count_ab + count_ba * 1j
        theta = np.angle(z)
        directionality = np.abs((4 * theta) / np.pi - 1)

        return directionality


class ChordAnalysis:

    def complexity(self):
        raise NotImplementedError

    def chromaticity(self):
        raise NotImplementedError


if __name__ == '__main__':
    metacorpora_path = 'petit_dcml_corpus/'
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    chord_transition_analysis = ChordTransitionAnalysis(metacorpora=metacorpora)

    bigram_tuples_sequential_M = chord_transition_analysis.chord_str_transitions_sequential(
        pieceinfo_list=metacorpora.get_annotated_pieces(),
        n=2, mode='M')
    bigram_tuples_sequential_m = chord_transition_analysis.chord_str_transitions_sequential(
        pieceinfo_list=metacorpora.get_annotated_pieces(),
        n=2, mode='m')

    bigrams = generics.Sequential.join([bigram_tuples_sequential_m, bigram_tuples_sequential_M])

    bigram_analysis = ChordBigramsAnalysis(bigram_tuples_sequential=bigrams)

