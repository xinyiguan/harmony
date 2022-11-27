#  By Xinyi Guan on 27 November 2022.

from modulation.representation import ModulationBigram
from modulation.plotting import Modulation
from modulation.loader import MetaCorpraInfo, CorpusInfo, PieceInfo

if __name__ == '__main__':
    meta_corpora_path = 'romantic_piano_corpus/'
    metacorpora = MetaCorpraInfo(meta_corpora_path=meta_corpora_path)
    modulation = Modulation(metacorpora)
    modulation.modulation_interval_by_modes_by_era(fig_path='fixed_figs/')
    print(modulation._modulation_df(data_source=metacorpora).to_markdown())
