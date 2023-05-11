import pandas as pd

from harmonytypes.numeral import Numeral
from stufentheorie import concepts
from harmonytypes.triad import Triad



def chord_chromaticism_vector():

    # 1. check chord type: whether or not numeral in the combined major-minor system or applied chord or neither


    # 2. if applied chord


    # 3. if in the major-minor system -> check mixture type


    # 4. if neither --> tonicization
    raise NotImplementedError

def test():
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')

    df_row = df.iloc[1]
    numeral = Numeral.from_df(df_row)
    print(f'{numeral=}')
    print(f'non-diatonic-notes: {numeral.non_diatonic_spcs(reference_key=numeral.key_if_tonicized())}')


if __name__ == "__main__":
    test()
