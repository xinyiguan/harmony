---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Chromaticity

```{code-cell} ipython3
from pitchtypes import asic, SpelledPitchClass, aspc, SpelledIntervalClass

from harmonytypes.degree import Degree
from harmonytypes.key import Key
from harmonytypes.numeral import Numeral
from harmonytypes.quality import TertianHarmonyQuality
from metrics import pc_content_index, within_chord_ci, within_key_ci, between_keys_ci
import os
from collections import defaultdict, Counter
from git import Repo
import dimcat
import ms3
import pandas as pd

from typing import Callable, Optional, TypeVar, Dict, Any

pd.set_option('display.max_columns', None)  # or 1000
concat_metadata_df = pd.read_csv('/Users/xinyiguan/Codes/musana/dcml_corpora/concatenated_metadata.tsv', sep='\t')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
CORPUS_PATH = os.environ.get('CORPUS_PATH', "/Users/xinyiguan/Codes/musana/dcml_corpora")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell} ipython3
repo = Repo(CORPUS_PATH)
notebook_repo = Repo('.', search_parent_directories=True)
```

```{code-cell} ipython3
dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH)
```

## helper functions

```{code-cell} ipython3
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


def get_expanded_dfs(data_set: dimcat.Dataset) -> Dict[Any, pd.DataFrame]:
    expanded_df_from_piece = lambda p: p.get_facet('expanded')[1]

    # check for eligible pieces

    dict_dfs = {k: expanded_df_from_piece(v) for k, v in data_set.pieces.items() if
                expanded_df_from_piece(v) is not None}
    return dict_dfs


def get_year_by_piecename(piece_name: str) -> int:
    year = concat_metadata_df[concat_metadata_df['fname'] == piece_name]['composed_end'].values[0]
    return year

def piece_wise_operation(piece_df: pd.DataFrame,
                         chord_wise_operation: Callable) -> pd.DataFrame:
    row_func = lambda row: pd.Series(maybe_bind(Numeral.from_df, chord_wise_operation)(row),
                                     dtype='object')

    mask = piece_df['chord'] != '@none'
    cleaned_piece_df = piece_df.dropna(subset=['chord']).reset_index(drop=True).loc[mask].reset_index(drop=True)
    result: pd.DataFrame = cleaned_piece_df.apply(row_func, axis=1)
    return result


def dataset_wise_operation(dataset: dimcat.Dataset,
                           piecewise_operation: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    expanded_df_dict = get_expanded_dfs(dataset)
    pieces_dict = {key: piecewise_operation(value) for key, value in expanded_df_dict.items() if
                   check_eligible_piece(value)}
    pieces_df: pd.DataFrame = pd.concat([v.assign(corpus=k[0], piece=k[1]) for k, v in pieces_dict.items()])
    return pieces_df
```

## Define chord-wise operations:

+++

### 1. pitch class content index

```{code-cell} ipython3
def pc_content_indices_dict(chord: Numeral) -> Dict[str, int]:
    global_key = chord.global_key
    local_key = chord.local_key
    key_if_tonicized = chord.key_if_tonicized()

    ndpc_GK = chord.non_diatonic_spcs(reference_key=global_key)
    pcci_GK_d5th_on_C = pc_content_index(numeral=chord, nd_ref_key=global_key, d5th_ref_tone=SpelledPitchClass("C"))
    pcci_GK_d5th_on_rt = pc_content_index(numeral=chord, nd_ref_key=global_key, d5th_ref_tone=global_key.tonic)

    ndpc_LK = chord.non_diatonic_spcs(reference_key=local_key)
    pcci_LK_d5th_on_C = pc_content_index(numeral=chord, nd_ref_key=local_key, d5th_ref_tone=SpelledPitchClass("C"))
    pcci_LK_d5th_on_rt = pc_content_index(numeral=chord, nd_ref_key=local_key, d5th_ref_tone=local_key.tonic)

    ndpc_TT = chord.non_diatonic_spcs(reference_key=key_if_tonicized)
    pcci_TT_d5th_on_C = pc_content_index(numeral=chord, nd_ref_key=key_if_tonicized,
                                         d5th_ref_tone=SpelledPitchClass("C"))
    pcci_TT_d5th_on_rt = pc_content_index(numeral=chord, nd_ref_key=key_if_tonicized,
                                          d5th_ref_tone=key_if_tonicized.tonic)

    result_dict = {'chord': chord.numeral_string,
                   'pcs': chord.spcs,
                   'global_tonic': global_key.to_string(),
                   'ndpc_GlobalTonic': ndpc_GK,
                   'm1_GlobalTonic': pcci_GK_d5th_on_C,
                   'm2_GlobalTonic': pcci_GK_d5th_on_rt,

                   'local_tonic': local_key.to_string(),
                   'ndpc_LocalTonic': ndpc_LK,
                   'm1_LocalTonic': pcci_LK_d5th_on_C,
                   'm2_LocalTonic': pcci_LK_d5th_on_rt,

                   'tonicized_key': key_if_tonicized.to_string(),
                   'ndpc_TonicizedTonic': ndpc_TT,
                   'm1_TonicizedTonic': pcci_TT_d5th_on_C,
                   'm2_TonicizedTonic': pcci_TT_d5th_on_rt}

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
```

### 2. multi-level chromatic index

```{code-cell} ipython3
def multilevel_ci_dict(chord: Numeral) -> Dict[str, int]:
    global_key = chord.global_key
    local_key = chord.local_key

    within_chord = within_chord_ci(numeral=chord)
    within_key = within_key_ci(reference_key=local_key, root=chord.root)
    between_key = between_keys_ci(source_key=global_key, target_key=local_key)

    result_dict = {"chord": chord.numeral_string,
                   "nd_notes": chord.non_diatonic_spcs(reference_key=chord.key_if_tonicized()),
                   "within_chord": within_chord,
                   "(LK,root)": (local_key, chord.root),
                   "within_key": within_key,
                   "(GK, LK)": (global_key, local_key),
                   "between_key": between_key}
    return result_dict
```

## Main

```{code-cell} ipython3
tchaikovsky = pd.read_csv(
    '/Users/xinyiguan/MusicData/dcml_corpora/tchaikovsky_seasons/harmonies/op37a12.tsv',
    sep='\t')

debussy = pd.read_csv(
    '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
    sep='\t')

result = piece_wise_operation(piece_df=debussy, chord_wise_operation=multilevel_ci_dict)
print(f'{result=}')
```

```{code-cell} ipython3

```
