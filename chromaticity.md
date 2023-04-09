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
import os
from collections import defaultdict, Counter

from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import typing
from analysis import ChromaticityMetrics
from dcml_corpora.utils import STD_LAYOUT, CADENCE_COLORS, chronological_corpus_order, color_background, get_repo_name, resolve_dir, value_count_df, get_repo_name, resolve_dir

pd.set_option('display.max_columns', None)  # or 1000
```

```{code-cell} ipython3
CORPUS_PATH = os.environ.get('CORPUS_PATH', "/Users/xinyiguan/Codes/musana/dcml_corpora")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
concat_metadata_df = pd.read_csv('/Users/xinyiguan/Codes/musana/dcml_corpora/concatenated_metadata.tsv', sep='\t')
```

```{code-cell} ipython3
repo = Repo(CORPUS_PATH)
notebook_repo = Repo('.', search_parent_directories=True)
print(f"Notebook repository '{get_repo_name(notebook_repo)}' @ {notebook_repo.commit().hexsha[:7]}")
print(f"Data repo '{get_repo_name(CORPUS_PATH)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
```

```{code-cell} ipython3
dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH)
```

## Chromaticity index

+++

### Helper functions

```{code-cell} ipython3
def get_expanded_dfs(data_set: dc.Dataset) -> typing.Dict[typing.Any, pd.DataFrame]:
    expanded_df_from_piece = lambda p: p.get_facet('expanded')[1]
    dict_dfs = {k: expanded_df_from_piece(v) for k, v in data_set.pieces.items() if
                expanded_df_from_piece(v) is not None}
    return dict_dfs


def get_year_by_piecename(piece_name: str) -> int:
    year = concat_metadata_df[concat_metadata_df['fname'] == piece_name]['composed_end'].values[0]
    return year


def piecewise_chromaticity_index1(piece_df: pd.DataFrame) -> pd.DataFrame:
    mask = piece_df['chord'] != '@none'
    cleaned_piece_df = piece_df.dropna(subset=['chord']).reset_index(drop=True).loc[mask].reset_index(drop=True)
    piece_chromaticity_index1_df = ChromaticityMetrics.metric1_df(piece=cleaned_piece_df).reset_index(drop=True)
    return piece_chromaticity_index1_df
```

### Main:

```{code-cell} ipython3
expanded_df_dict = get_expanded_dfs(dataset)
pieces_dict = {key: piecewise_chromaticity_index1(value) for key, value in expanded_df_dict.items()}
pieces_df: pd.DataFrame = pd.concat([v.assign(corpus=k[0], piece=k[1]) for k, v in pieces_dict.items()])

pieces_df['year'] = [get_year_by_piecename(name) for name in pieces_df['piece']]
```

```{code-cell} ipython3
pieces_df
```

```{code-cell} ipython3
column_order = ['chord','pcs', 'global_tonic','d_to_global', 'local_tonic', 'd_to_local','corpus','piece', 'year' ]

result_df = pieces_df.reindex(columns=column_order)
result_df
```

```{code-cell} ipython3
result_df.head(30)
```

# Plottings

```{code-cell} ipython3
fig = px.scatter(pieces_df, x='year', y=['d_to_local', 'd_to_global'], labels={
                  'value': 'Differences', 'year': 'Year'})
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

### Test

```{code-cell} ipython3
# find index of strange chords
pd.set_option('display.max_columns', None)  # or 1000

the_expanded_dict = get_expanded_dfs(dataset)
the_expanded_df = pd.concat([v.assign(corpus=k[0], piece=k[1]) for k, v in the_expanded_dict.items()])
the_expanded_df = the_expanded_df.reset_index(drop=True)
```

```{code-cell} ipython3
pd.set_option('display.max_columns', None)  # or 1000

row = the_expanded_df.loc[the_expanded_df['chord']=='ii+2']
row
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
expanded_df = get_expanded_dfs(dataset)
```

```{code-cell} ipython3
subdict={a:expanded_df[a] for a in [('ABC','n08op59-2_03'),
                                    ('corelli','op03n12f'),
                                    ('grieg_lyric_pieces','op62n02')]}

subdict
```

```{code-cell} ipython3
sub_pieces_dict = {key: piecewise_chromaticity_index1(value.dropna(subset=['chord'])[value.chord !='@none']) for key, value in subdict.items()}
sub_pieces_dict
```

```{code-cell} ipython3
sub_pieces_df = pd.concat([v.assign(corpus=k[0], piece=k[1]) for k, v in sub_pieces_dict.items()])
sub_pieces_df
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
pieces_dict = {key: piecewise_chords_per_beat(value) for key, value in get_expanded_dfs(dataset).items()}
pieces_df = pd.Series(pieces_dict, name='harmonic_density').to_frame()
pieces_df['year'] = [get_year_by_piecename(name) for name in pieces_df.index.get_level_values(1)]
pieces_df.index.set_names(['corpus', 'piece'], inplace=True)  # set names for MultiIndex
```

### Plottings

```{code-cell} ipython3
# prepare the df for scatter plot

flatten_df = pieces_df.reset_index()
flatten_df
```

```{code-cell} ipython3
fig = px.scatter(flatten_df, x="year", y="harmonic_density", color="corpus", hover_data=['piece'], opacity=0.5)
fig.show()
```

```{code-cell} ipython3

```
