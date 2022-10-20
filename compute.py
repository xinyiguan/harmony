from typing import Union, Literal, List

import numpy as np
import pandas as pd
from scipy.stats import entropy

from loader import PieceInfo, CorpusInfo, MetaCorpraInfo
from n_gram import N_Gram


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
            corpus = CorpusInfo(obj.parent_corpus_path)
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


def compute_information_content(probability: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the information content (self-information) of indiviudal event with base 2 (in bits)
    :param probability: a DataFrame of probabilities with the corresponding key values as row index.
    :return: a DataFrame of ic with the corresponding key values as row index.
    """
    h = -np.log2(probability)
    h.columns = ['information content']
    return h


def compute_entropy(probability: pd.DataFrame, base: int = 2) -> float:
    H = entropy(probability, base=base)
    return H


def compute_bigram_probs(obj: Union[PieceInfo, CorpusInfo, MetaCorpraInfo],
                         aspect: Literal['harmonies', 'measures', 'notes'],
                         key: str,
                         meta_corpora_path: str = None):
    """
    compute the bigram probability: P(c2|c1) = count(c1 c2) / count(c1)
    :param obj:
    :param aspect:
    :param key:
    :param meta_corpora_path:
    :return:
    """
    if isinstance(obj, PieceInfo):
        unigrams = pd.DataFrame(N_Gram(piece=obj, aspect=aspect, key=key, n=1).get_grams_seq(), columns=['unigram'])
        bigrams = pd.DataFrame(N_Gram(piece=obj, aspect=aspect, key=key, n=2).get_grams_seq(), columns=['bigram'])

        unigram_counts = unigrams['unigram'].value_counts(sort=True)
        bigram_counts = bigrams['bigram'].value_counts(sort=True)
        correpond_preceding_unigram = [name.split('_')[0] for name in bigram_counts.index.values.flatten().tolist()]

        bigram_probs = pd.DataFrame(bigram_counts.index.values.flatten(), columns=['bigram_type'])
        bigram_probs['preceding_unigram']= correpond_preceding_unigram
        bigram_probs['probs'] = bigram_counts[bigram_probs['bigram_type']].values/unigram_counts[bigram_probs['preceding_unigram']].values
        return bigram_probs

    elif isinstance(obj, CorpusInfo):
        corpus_unigrams_df = pd.DataFrame(obj.get_corpus_aspect_df(aspect=aspect, selected_keys=[key]).values.flatten().tolist(), columns=['unigram'])
        corpus_bigrams_list = []

        for idx, piece in enumerate(obj.piece_list):
            bigrams = pd.DataFrame(N_Gram(piece=piece, aspect=aspect, key=key, n=2).get_grams_seq(), columns=['bigram'])
            corpus_bigrams_list.append(bigrams)

        corpus_bigrams_df = pd.concat(corpus_bigrams_list, ignore_index=True)

        corpus_unigram_counts =corpus_unigrams_df['unigram'].value_counts()
        corpus_bigram_counts = corpus_bigrams_df['bigram'].value_counts()
        correpond_preceding_unigram = [name.split('_')[0] for name in corpus_bigram_counts.index.values.flatten().tolist()]

        bigram_probs = pd.DataFrame(corpus_bigram_counts.index.values.flatten(), columns=['bigram_type'])
        bigram_probs['preceding_unigram']= correpond_preceding_unigram
        bigram_probs['probs'] = corpus_bigram_counts[bigram_probs['bigram_type']].values/corpus_unigram_counts[bigram_probs['preceding_unigram']].values
        return bigram_probs

    elif isinstance(obj, MetaCorpraInfo):
        raise NotImplementedError



if __name__ == '__main__':
    metacorpora_path = 'romantic_piano_corpus/'
    corpus_path = 'romantic_piano_corpus/debussy_suite_bergamasque/'

    metacorpora = MetaCorpraInfo(metacorpora_path)
    corpus = CorpusInfo(corpus_path)

    compute_bigram_probs(obj=corpus, aspect='harmonies', key='numeral')
