from typing import Union, Literal, List

import pandas as pd

from loader import PieceInfo, CorpusInfo, MetaCorpraInfo


def dataframe_extension(source_df: pd.DataFrame,
                        extended_unique_kw_list: List[str],
                        fillNaN: bool) -> pd.DataFrame:
    """
    This function takes one dataframe and a list.
    1) a dataframe of keyword and corresponding values.
    2) a list of extended unique keyword.
    Need to use the extended unique keyword list as the row index, and then fill in the correponding values. If nan, fill 0.
    """
    target_df = pd.DataFrame(index=extended_unique_kw_list, columns=[source_df.columns.values.tolist()[0]])
    for idx, val in enumerate(source_df.index.values.tolist()):
        target_df.loc[val] = source_df.loc[val]
    if fillNaN:
        target_df = target_df.fillna(0)
        return target_df
    else:
        return target_df


def compute_prob(obj: Union[PieceInfo, CorpusInfo, MetaCorpraInfo],
                 aspect: Literal['harmonies', 'measures', 'notes'],
                 key: str,
                 keyword_normalization: Literal['within_piece', 'within_corpus', 'within_metacorpora'],
                 meta_corpora_path: str = None) -> pd.DataFrame:
    """
    Compute the probabilities of each key value, with the key value as the row index.
    Note that we account for all unique keys (in either piece/corpus/metacorpora level) as row index to calculate probs.
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
        probs = probs.set_axis(['probs'], axis='columns')
        if keyword_normalization == 'within_piece':
            return probs
        elif keyword_normalization == 'within_corpus':
            source_df = probs
            corpus = CorpusInfo(piece.parent_corpus_path)
            extended_unique_kw_list = corpus.get_corpuswise_unique_key_values(aspect=aspect, key=key)
            extended_probs = dataframe_extension(source_df=source_df, extended_unique_kw_list=extended_unique_kw_list,
                                                 fillNaN=True)
            return extended_probs

        elif keyword_normalization == 'within_metacorpora':
            if metacorpora_path is None:
                raise AssertionError(f'{meta_corpora_path} Missing required input')
            else:
                metacorpora = MetaCorpraInfo(meta_corpora_path=metacorpora_path)
                meta_unique_key_vals = metacorpora.get_corpora_unique_key_values(aspect=aspect, key=key)
                meta_extended_probs = dataframe_extension(source_df=probs, extended_unique_kw_list=meta_unique_key_vals,
                                                          fillNaN=True)
                return meta_extended_probs

    elif isinstance(obj, CorpusInfo):
        corpus_df = obj.get_corpus_aspect_df(aspect=aspect, selected_keys=[key]).copy()
        unique_key_vals = obj.get_corpuswise_unique_key_values(aspect=aspect, key=key)
        counts = corpus_df[key].value_counts()
        probs = counts / counts.sum()
        probs = probs.to_frame()
        probs = probs.set_axis(['probs'], axis='columns')
        extended_probs = dataframe_extension(source_df=probs, extended_unique_kw_list=unique_key_vals,
                                             fillNaN=True)
        if keyword_normalization == 'within_corpus':
            return extended_probs
        elif keyword_normalization == 'within_metacorpora':
            if metacorpora_path is None:
                raise AssertionError(f'{meta_corpora_path} Missing required input')
            else:
                metacorpora = MetaCorpraInfo(meta_corpora_path=metacorpora_path)
                meta_unique_key_vals = metacorpora.get_corpora_unique_key_values(aspect=aspect, key=key)
                meta_extended_probs = dataframe_extension(source_df=extended_probs,
                                                          extended_unique_kw_list=meta_unique_key_vals, fillNaN=True)
                return meta_extended_probs

    elif isinstance(obj, MetaCorpraInfo):
        metacorpora_df = obj.get_corpora_aspect_df(aspect=aspect, selected_keys=[key]).copy()
        counts = metacorpora_df[key].value_counts()
        probs = counts / counts.sum()
        probs = probs.to_frame()
        probs = probs.set_axis(['probs'], axis='columns')
        return probs


if __name__ == '__main__':
    metacorpora_path = 'romantic_piano_corpus/'
    corpus_path = 'romantic_piano_corpus/debussy_suite_bergamasque/'

    metacorpora = MetaCorpraInfo(metacorpora_path)
    corpus = CorpusInfo(corpus_path)

    piece = PieceInfo(parent_corpus_path=corpus_path, piece_name='l075-01_suite_prelude')
    # within_piece = compute_prob(obj=piece, aspect='harmonies', key='numeral', keyword_normalization='within_piece')
    # print(within_piece)
    within_corp = compute_prob(obj=corpus, aspect='harmonies', key='numeral',
                               keyword_normalization='within_metacorpora', meta_corpora_path=metacorpora_path)
    print(within_corp)
    # print(prob_df.index.values.tolist())
    # print(prob_df.loc['V'])

    # source_df = prob_df
    # extended_unique_kw_list = metacorpora.get_corpora_unique_key_values(aspect='harmonies', key='numeral')
    #
    # print(dataframe_extension(source_df=source_df , extended_unique_kw_list= extended_unique_kw_list, fillNaN=True))
