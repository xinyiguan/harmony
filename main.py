from pitchtypes import asic, SpelledPitchClass, aspc

from harmonytypes.degree import Degree
from harmonytypes.key import Key
from harmonytypes.numeral import Numeral
from harmonytypes.quality import TertianHarmonyQuality
from metrics import pc_content_index_m1, pc_content_index_m2
import os
from collections import defaultdict, Counter
from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Callable, Optional, TypeVar, Dict, Any
from dcml_corpora.utils import STD_LAYOUT, CADENCE_COLORS, chronological_corpus_order, color_background, get_repo_name, \
    resolve_dir, value_count_df, get_repo_name, resolve_dir

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def maybe_bind(f: Callable[[A], Optional[B]], g: Callable[[B], Optional[C]]) -> Callable[[A], Optional[C]]:
    def h(x: A) -> Optional[C]:
        y = f(x)
        if y is None:
            return None
        else:
            return g(y)

    return h





# helper functions ==========================================
def check_eligible_piece(piece_df: pd.DataFrame) -> bool:
    """
    Check if the specified columns in a pandas DataFrame contain missing values.
    Returns True if all columns have all values, False otherwise.
    """
    columns = ["chord", "chord_tones", "root", "bass_note"]
    missing_values = piece_df[columns].isnull().any()
    if missing_values.any():
        return False
    else:
        return True

def get_expanded_dfs(data_set: dc.Dataset) -> Dict[Any, pd.DataFrame]:
    expanded_df_from_piece = lambda p: p.get_facet('expanded')[1]

    # check for eligible pieces

    dict_dfs = {k: expanded_df_from_piece(v) for k, v in data_set.pieces.items() if
                expanded_df_from_piece(v) is not None}
    return dict_dfs


def get_year_by_piecename(piece_name: str) -> int:
    year = concat_metadata_df[concat_metadata_df['fname'] == piece_name]['composed_end'].values[0]
    return year




# ===============================================================

def pc_content_indices_dict(chord: Numeral) -> Dict[str, int]:
    global_key = chord.global_key
    local_key = chord.local_key
    key_if_tonicized = chord.key_if_tonicized()

    ndpc_GlobalTonic = chord.non_diatonic_spcs(reference_key=global_key)
    pcci_GlobalTonic_m1 = pc_content_index_m1(numeral=chord, reference_key=global_key)
    pcci_GlobalTonic_m2 = pc_content_index_m2(numeral=chord, reference_key=global_key)

    ndpc_LocalTonic = chord.non_diatonic_spcs(reference_key=local_key)
    pcci_LocalTonic_m1 = pc_content_index_m1(numeral=chord, reference_key=local_key)
    pcci_LocalTonic_m2 = pc_content_index_m2(numeral=chord, reference_key=local_key)

    ndpc_TonicizedTonic = chord.non_diatonic_spcs(reference_key=key_if_tonicized)
    pcci_TonicizedTonic_m1 = pc_content_index_m1(numeral=chord, reference_key=key_if_tonicized)
    pcci_TonicizedTonic_m2 = pc_content_index_m2(numeral=chord, reference_key=key_if_tonicized)

    result_dict = {'chord': chord.numeral_string,
                   'pcs': chord.spcs,
                   'global_tonic': global_key.to_string(),
                   'ndpc_GlobalTonic': ndpc_GlobalTonic,
                   'm1_GlobalTonic': pcci_GlobalTonic_m1,
                   'm2_GlobalTonic': pcci_GlobalTonic_m2,

                   'local_tonic': local_key.to_string(),
                   'ndpc_LocalTonic': ndpc_LocalTonic,
                   'm1_LocalTonic': pcci_LocalTonic_m1,
                   'm2_LocalTonic': pcci_LocalTonic_m2,

                   'tonicized_key': key_if_tonicized.to_string(),
                   'ndpc_TonicizedTonic': ndpc_TonicizedTonic,
                   'm1_TonicizedTonic': pcci_TonicizedTonic_m1,
                   'm2_TonicizedTonic': pcci_TonicizedTonic_m2}

    return result_dict


def piecewise_pc_content_indices_df(piece_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with columns: chord, pcs, gt, ndpc_gt, m1_gt, m2_gt,
    lt, ndpc_lt, m1_lt, m2_lt, tk, ndpc_tt, m1_tt, m2_tt for a piece.
    """
    row_func = lambda row: pd.Series(maybe_bind(Numeral.from_df, pc_content_indices_dict)(row),
                                     dtype='object')

    mask = piece_df['chord'] != '@none'
    cleaned_piece_df = piece_df.dropna(subset=['chord']).reset_index(drop=True).loc[mask].reset_index(drop=True)
    result: pd.DataFrame = cleaned_piece_df.apply(row_func, axis=1)
    return result




def test():
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')

    df_row = df.iloc[1]
    numeral = Numeral.from_df(df_row)

    result = pc_content_indices_dict(chord=numeral)
    print(f'{result=}')


def test2():
    tchaikovsky = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/tchaikovsky_seasons/harmonies/op37a12.tsv',
        sep='\t')

    debussy = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')
    result = piecewise_pc_content_indices_df(piece_df=tchaikovsky)
    print(f'{result=}')


if __name__ == '__main__':
    CORPUS_PATH = os.environ.get('CORPUS_PATH', "/Users/xinyiguan/Codes/musana/dcml_corpora")
    CORPUS_PATH = resolve_dir(CORPUS_PATH)
    concat_metadata_df = pd.read_csv('/Users/xinyiguan/Codes/musana/dcml_corpora/concatenated_metadata.tsv', sep='\t')

    # repo = Repo(CORPUS_PATH)
    # notebook_repo = Repo('.', search_parent_directories=True)

    dataset = dc.Dataset()
    dataset.load(directory=CORPUS_PATH)

    expanded_df_dict = get_expanded_dfs(dataset)
    pieces_dict = {key: piecewise_pc_content_indices_df(value) for key, value in expanded_df_dict.items() if check_eligible_piece(value)}
    pieces_df: pd.DataFrame = pd.concat([v.assign(corpus=k[0], piece=k[1]) for k, v in pieces_dict.items()])

    print(pieces_df)

