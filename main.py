# Created by Xinyi Guan in 2022.
import glob
import os

import pandas as pd


# 1. chord transitions and chord triplets (3-grams) and their information-theoretic properties

# 1.  entropy of local key (target key of modulation)

def assemble_localkey_entropy_df(metacorpora_path: str):
    corpusname_list = sorted([f for f in os.listdir(metacorpora_path)
                              if not f.startswith('.')
                              if not f.startswith('__')])

    harmonies_paths_list = [metacorpora_path + item + '/harmonies/' for item in corpusname_list]
    metacorpora_harmonies_df_list = []
    for path in harmonies_paths_list:
        corpus_df = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(path + '*.tsv')])
        metacorpora_harmonies_df_list.append(corpus_df)

    metacorpora_harmonies_df = pd.concat(metacorpora_harmonies_df_list)

    return metacorpora_harmonies_df


def plot_localkey_entropy_by_pieces(data, hue_by):
    pass


# 2.  entropy of chord vocab

if __name__ == '__main__':
    result = assemble_localkey_entropy_df(metacorpora_path='romantic_piano_corpus/')
    print(result)
