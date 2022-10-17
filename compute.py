from typing import Union, Literal

import pandas as pd

from loader import PieceInfo, CorpusInfo, MetaCorpraInfo


def compute_prob(obj: Union[PieceInfo, CorpusInfo, MetaCorpraInfo],
                 aspect: Literal['harmonies', 'measures', 'notes'],
                 key: str) -> pd.DataFrame:
    """

    :param: obj:
    :param: aspect:
    :param: key:

    :return: a DataFrame, with the key value as index, and prob as val
    """
    if isinstance(obj, PieceInfo):
        piece_df = obj.get_aspect_df(aspect=aspect, selected_keys=[key]).copy()
        counts = piece_df[key].value_counts()
        probs = counts / counts.sum()
        probs = probs.to_frame()
        return probs


if __name__ == '__main__':
    metacorpora_path = 'romantic_piano_corpus/'
    corpus_path = 'romantic_piano_corpus/debussy_suite_bergamasque/'

    piece = PieceInfo(parent_corpus_path=corpus_path, piece_name='l075-01_suite_prelude')
    print(compute_prob(obj=piece, aspect='harmonies', key='numeral'))
