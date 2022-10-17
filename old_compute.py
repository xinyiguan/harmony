import os
from typing import List

import dimcat
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipyentropy


def get_annotated_corpus(corpus: dimcat.data.Corpus) -> pd.DataFrame:
    annotated: dimcat.data.Corpus = dimcat.filter.IsAnnotatedFilter().process_data(corpus)
    annotated_corpus = annotated.get_facet(what='expanded')
    return annotated_corpus


def count(corpus: dimcat.Corpus, key: str) -> pd.DataFrame:
    annotated_corpus = get_annotated_corpus(corpus)
    data_df = annotated_corpus[key]
    data_count = data_df.value_counts()
    return data_count


def compute_probs(corpus: dimcat.Corpus, key: str) -> pd.DataFrame:
    """Compute the probs"""
    counts = count(corpus=corpus, key=key)
    probs = counts / counts.sum()
    return probs


def compute_information_content(corpus: dimcat.Corpus, key: str) -> pd.DataFrame:
    """Compute the information content (self-information) of indiviudal event with base 2 (in bits) """
    counts = count(corpus=corpus, key=key)
    probs = counts / counts.sum()
    # print(probs)
    h = -np.log2(probs)
    return h


def compute_entropy(corpus: dimcat.Corpus, key: str) -> float:
    """Compute the entropy of a distribution"""
    counts = count(corpus=corpus, key=key)
    probs = counts / counts.sum()
    H = -sum([p * np.log2(p) for idx, p in enumerate(probs)])
    return H


def compute_entropy_scipy(corpus: dimcat.Corpus, key: str, base: int = 2) -> float:
    """Compute the entropy of a distribution, using scipy library"""
    counts = count(corpus=corpus, key=key)
    probs = counts / counts.sum()
    H = scipyentropy(probs, base=base)
    return H


def get_subcorpus_path(metacorpora_path: str) -> List[str]:
    subcorpus_list = [metacorpora_path + f + '/' for f in os.listdir(metacorpora_path) if not f.startswith('.')]
    return subcorpus_list


def compute_corpus_mean_composition_year(corpus: dimcat.Corpus):
    composition_yrs_list = corpus.data.metadata(from_tsv=True).composed_end.to_list()
    mean_composition_year = int(np.mean(composition_yrs_list))
    return mean_composition_year


#
# def get_unique_chord_labels_from_corpora_collection(metacorpora_path: str) -> List[str]:
#     """filtered with annotated corpus"""
#     subcorpus_list = get_subcorpus_path(metacorpora_path)
#     for idx, corpus_path in subcorpus_list:
#         corpus = dimcat.data.Corpus().load(directory=[corpus_path])
#         corpus=get_annotated_corpus(corpus)
#
#
#     return chord_label_list


if __name__ == '__main__':
    corpus_path = 'romantic_piano_corpus/chopin_mazurkas/'
    corpus = dimcat.data.Corpus()
    corpus.load(directory=[corpus_path])
    year = compute_corpus_mean_composition_year(corpus)
