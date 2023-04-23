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
import dimcat
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Callable
from dcml_corpora.utils import STD_LAYOUT, CADENCE_COLORS, chronological_corpus_order, color_background, get_repo_name, resolve_dir, value_count_df, get_repo_name, resolve_dir

from harmonytypes.numeral import Numeral, NumeralChain
from main import maybe_bind, pc_content_indices_dict, piecewise_pc_content_indices_df
from metrics import pc_content_index_m1, pc_content_index_m2

pd.set_option('display.max_columns', None)  # or 1000
```

```{code-cell} ipython3

```

```{code-cell} ipython3
CORPUS_PATH = os.environ.get('CORPUS_PATH', "/Users/xinyiguan/Codes/musana/dcml_corpora")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
concat_metadata_df = pd.read_csv('/Users/xinyiguan/Codes/musana/dcml_corpora/concatenated_metadata.tsv', sep='\t')
```

```{code-cell} ipython3
repo = Repo(CORPUS_PATH)
notebook_repo = Repo('.', search_parent_directories=True)
```

```{code-cell} ipython3
dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH)
```

## Main

```{code-cell} ipython3
result = dataset_wise_operation(dataset=dataset, piecewise_operation=piecewise_pc_content_indices_df)
```

```{code-cell} ipython3
result
```

```{code-cell} ipython3

```
