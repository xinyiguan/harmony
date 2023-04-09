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
from dcml_corpora.utils import STD_LAYOUT, CADENCE_COLORS, chronological_corpus_order, color_background, get_repo_name, \
    resolve_dir, value_count_df, get_repo_name, resolve_dir

CORPUS_PATH = os.environ.get('CORPUS_PATH', "/Users/xinyiguan/Codes/musana/dcml_corpora")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
concat_metadata_df = pd.read_csv('/Users/xinyiguan/Codes/musana/dcml_corpora/concatenated_metadata.tsv', sep='\t')

repo = Repo(CORPUS_PATH)
notebook_repo = Repo('.', search_parent_directories=True)

dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH)


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


if __name__ == '__main__':
    expanded_df_dict = get_expanded_dfs(dataset)
    pieces_dict = {key: piecewise_chromaticity_index1(value) for key, value in expanded_df_dict.items()}
    pieces_df: pd.DataFrame = pd.concat([v.assign(corpus=k[0], piece=k[1]) for k, v in pieces_dict.items()])
    pieces_df['year'] = [get_year_by_piecename(name) for name in pieces_df['piece']]
    print(pieces_df)
    #
    # fig = px.scatter(pieces_df, x="year", y="harmonic_density", color="corpus", hover_data=['piece'],symbol="species", opacity=0.5)
    # fig.show()