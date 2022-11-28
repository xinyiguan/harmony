# Created by Xinyi Guan in 2022.

from modulation.representation import ModulationBigram
from modulation.plotting import Modulation
from modulation.loader import MetaCorpraInfo, CorpusInfo, PieceInfo

if __name__ == '__main__':
    meta_corpora_path = 'dcml_corpora/'
    metacorpora = MetaCorpraInfo(meta_corpora_path=meta_corpora_path)
    modulation = Modulation(metacorpora)
    modulation.plot_chronological_distribution_of_pieces(fig_path='fixed_figs_bigram/', hue_by='corpus', fig_name='chronological_distr_corpus')
    modulation.plot_chronological_distribution_of_pieces(fig_path='fixed_figs_bigram/', hue_by='mode', fig_name='chronological_distr_mode')

    # print(modulation._target_modulation_df(data_source=metacorpora).to_markdown())
    # df = modulation._modulation_bigram_df(data_source=metacorpora)
    # print(df.loc[(df['interval'] == 0) & (df['era'] == "Baroque")].to_markdown())
