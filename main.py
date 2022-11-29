# Created by Xinyi Guan in 2022.
# A meta-script for all plots and computations for the report
from typing import Dict

import plotting
from modulation.representation import ModulationBigram, HarmonicProgression
from modulation.plotting import Modulation
from modulation.loader import MetaCorpraInfo, CorpusInfo, PieceInfo


# _______________________________________ plots _________________________________________

# 1. metacorpora_basics:
def plot_metacorpora_basics(metacorpora: MetaCorpraInfo, fig_path: str):
    modulation = Modulation(metacorpora)
    modulation.plot_chronological_distribution_of_pieces(fig_path=fig_path, hue_by=None, fig_name='chronological_distr')
    modulation.plot_chronological_distribution_of_pieces(fig_path=fig_path, hue_by='corpus',
                                                         fig_name='chronological_distr_corpus')
    modulation.plot_chronological_distribution_of_pieces(fig_path=fig_path, hue_by='mode',
                                                         fig_name='chronological_distr_mode')


# 2. Frequency of modulations in the corpus
def plot_modulation_frequencies(metacorpora: MetaCorpraInfo, fig_path: str):
    modulation = Modulation(metacorpora)
    modulation.plot_modulation_counts_in_a_piece(fig_path=fig_path)


# 3. Modulation steps(interval)
def plot_modulation_steps(metacorpora: MetaCorpraInfo, fig_path: str):
    modulation = Modulation(metacorpora)
    modulation.plot_modulation_interval_distr_by_modes(fig_path=fig_path)
    modulation.plot_modulation_interval_by_era(fig_path=fig_path)
    modulation.plot_modulation_interval_by_modes_by_era(fig_path=fig_path)
    modulation.plot_violinplot_swarmplot_by_era(fig_path=fig_path)


# _______________________________________ compute _________________________________________
def weight_of_key_region(piece: PieceInfo) -> Dict:
    key_region_weights_dict = {}
    total_labels_num = piece.get_aspect_df(aspect='harmonies', selected_keys=None).shape[0]
    key_region_subdfs_list = piece.get_key_region_subdfs_list()
    for item in key_region_subdfs_list:
        hamonic_progression = HarmonicProgression.parse(key_region_df=item)
        subdf_label_num = hamonic_progression.length()
        proportion = subdf_label_num / total_labels_num
        key_region_weights_dict[hamonic_progression.localkey.numeral_str] = proportion
    return key_region_weights_dict


if __name__ == '__main__':
    # # Define the working metacorpora:
    meta_corpora_path = 'romantic_piano_corpus/'
    metacorpora = MetaCorpraInfo(meta_corpora_path=meta_corpora_path)
    modulation = Modulation(metacorpora)
    result = modulation.localkey_region_profile_df()['era']
    print(result)
