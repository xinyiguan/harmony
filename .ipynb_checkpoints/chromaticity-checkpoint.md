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

```{code-cell} ipython3
result
```

```{code-cell} ipython3

```
