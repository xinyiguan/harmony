import pandas as pd

from harmonytypes.numeral import Numeral


def rootChromaticism(a: Numeral) -> str:
    root = a.root


def thirdChromaticism(chord_annoatation: Numeral) -> str:
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
