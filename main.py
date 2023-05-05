import os

import numpy as np
from pitchtypes import asic, SpelledPitchClass, aspc, SpelledIntervalClass

from harmonytypes.degree import Degree
from harmonytypes.key import Key
from harmonytypes.numeral import Numeral
from harmonytypes.quality import TertianHarmonyQuality
from harmonytypes.util import maybe_bind
from metrics import pc_content_index, ChromaticIndex_Def2
from collections import defaultdict, Counter
from git import Repo
import dimcat
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

import matplotlib.colors as mc
import matplotlib.image as image
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable import cartocolors
from typing import Callable, Optional, TypeVar, Dict, Any, Tuple, Literal
from dcml_corpora.utils import STD_LAYOUT, CADENCE_COLORS, chronological_corpus_order, color_background, get_repo_name, \
    resolve_dir, value_count_df, get_repo_name, resolve_dir


# helper functions: Data preprocessing ==========================================

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


def get_year_by_piecename(piece_name: str,
                          meatadata_tsv_path: str = '/Users/xinyiguan/Codes/musana/dcml_corpora/concatenated_metadata.tsv') -> int:
    concat_metadata_df = pd.read_csv(meatadata_tsv_path, sep='\t')
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
                           piecewise_operation: Callable) -> pd.DataFrame:
    expanded_df_dict = get_expanded_dfs(dataset)
    pieces_dict = {key: piecewise_operation(value) for key, value in expanded_df_dict.items() if
                   check_eligible_piece(value)}

    pieces_df: pd.DataFrame = pd.concat([v.assign(corpus=k[0], piece=k[1]) for k, v in pieces_dict.items()])

    pieces_df['year'] = [get_year_by_piecename(name) for name in pieces_df["piece"]]
    # pieces_df.index.set_names(['corpus', 'piece'], inplace=True)  # set names for MultiIndex
    return pieces_df


# ===============================================================

def CI1_pc_content_indices_dict(chord: Numeral) -> Dict[str, int]:
    global_key = chord.global_key
    local_key = chord.local_key
    key_if_tonicized = chord.key_if_tonicized()

    ndpc_GK = [str(x) for x in chord.non_diatonic_spcs(reference_key=global_key)]
    pcci_GK_d5th_on_C = pc_content_index(numeral=chord, nd_ref_key=global_key, d5th_ref_tone=SpelledPitchClass("C"))
    pcci_GK_d5th_on_rt = pc_content_index(numeral=chord, nd_ref_key=global_key, d5th_ref_tone=global_key.tonic)

    ndpc_LK = [str(x) for x in chord.non_diatonic_spcs(reference_key=local_key)]
    pcci_LK_d5th_on_C = pc_content_index(numeral=chord, nd_ref_key=local_key, d5th_ref_tone=SpelledPitchClass("C"))
    pcci_LK_d5th_on_rt = pc_content_index(numeral=chord, nd_ref_key=local_key, d5th_ref_tone=local_key.tonic)

    ndpc_TT = [str(x) for x in chord.non_diatonic_spcs(reference_key=key_if_tonicized)]
    pcci_TT_d5th_on_C = pc_content_index(numeral=chord, nd_ref_key=key_if_tonicized,
                                         d5th_ref_tone=SpelledPitchClass("C"))
    pcci_TT_d5th_on_rt = pc_content_index(numeral=chord, nd_ref_key=key_if_tonicized,
                                          d5th_ref_tone=key_if_tonicized.tonic)

    result_dict = {'chord': chord.numeral_string,
                   'pcs': chord.spcs,
                   'global_key': global_key.to_string(),
                   'ndpc_GlobalTonic': ndpc_GK,
                   'm1_GlobalTonic': pcci_GK_d5th_on_C,
                   'm2_GlobalTonic': pcci_GK_d5th_on_rt,

                   'local_key': local_key.to_string(),
                   'ndpc_LocalTonic': ndpc_LK,
                   'm1_LocalTonic': pcci_LK_d5th_on_C,
                   'm2_LocalTonic': pcci_LK_d5th_on_rt,

                   'tonicized_key': key_if_tonicized.to_string(),
                   'ndpc_TonicizedTonic': ndpc_TT,
                   'm1_TonicizedTonic': pcci_TT_d5th_on_C,
                   'm2_TonicizedTonic': pcci_TT_d5th_on_rt}

    return result_dict


def CI1_piecewise_pc_content_indices_df(piece_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with columns: chord, pcs, gt, ndpc_gt, m1_gt, m2_gt,
    lt, ndpc_lt, m1_lt, m2_lt, tk, ndpc_tt, m1_tt, m2_tt for a piece.

    m1 corresponds to the distant on the line of fifths centered on C
    m2 corresponds to the distant on the line of fifths centered on the root of the chord
    """

    result = piece_wise_operation(piece_df=piece_df, chord_wise_operation=CI1_pc_content_indices_dict)
    return result


# ===============================================================

def CI2_multilevel_ci_dict(chord: Numeral) -> Dict[str, int]:
    global_key = chord.global_key
    local_key = chord.local_key

    within_chord = ChromaticIndex_Def2.within_chord_ci(numeral=chord)
    within_key = ChromaticIndex_Def2.within_key_ci(reference_key=local_key, root=chord.root)
    between_key = ChromaticIndex_Def2.between_keys_ci(source_key=global_key, target_key=local_key)

    result_dict = {"chord": chord.numeral_string,
                   "nd_notes": chord.non_diatonic_spcs(reference_key=chord.key_if_tonicized()),
                   "within_chord": within_chord,
                   "(LK,root)": (local_key, chord.root),
                   "within_key": within_key,
                   "(GK, LK)": (global_key, local_key),
                   "between_key": between_key}
    return result_dict


# ===============================================================
class DataframePrep:
    @staticmethod
    def CI1_pc_content_index_df() -> pd.DataFrame:
        CORPUS_PATH = os.environ.get('CORPUS_PATH', "/Users/xinyiguan/Codes/musana/dcml_corpora")
        CORPUS_PATH = resolve_dir(CORPUS_PATH)

        mydataset = dimcat.Dataset()
        mydataset.load(directory=CORPUS_PATH)

        dataset_df = dataset_wise_operation(dataset=mydataset,
                                            piecewise_operation=CI1_piecewise_pc_content_indices_df)
        dataset_df.to_csv("temp_dataframes/CI1_corpus_pc_content_df", sep="\t")
        return dataset_df

    @staticmethod
    def CI1_lollipop_pc_content_index_m2l_df() -> Tuple[str, pd.DataFrame]:
        dataset_df = pd.read_csv("temp_dataframes/CI1_corpus_pc_content_df", sep='\t')
        plot_ready_df = dataset_df.groupby("piece").agg(corpus=("corpus", "first"), year=("year", "first"),
                                                        chord_num=("chord", "count"), max_val=("m2_LocalTonic", "max"),
                                                        min_val=("m2_LocalTonic", "min"),
                                                        pieces_avg=("m2_LocalTonic", np.mean)).reset_index()

        plot_ready_df = plot_ready_df.assign(
            corpus_year=plot_ready_df.groupby("corpus")["year"].transform(np.mean),
            # take the mean of the pieces year in a corpus
            corpus_avg=plot_ready_df.groupby("corpus")["pieces_avg"].transform(np.mean)).sort_values(
            ["corpus_year", "year"]).reset_index()

        corpus_dict = {corpus: i + 1 for i, corpus in
                       enumerate(plot_ready_df.sort_values(["corpus_year", "year"])['corpus'].unique())}
        piece_dict = {piece: i + 1 for i, piece in
                      enumerate(plot_ready_df.sort_values(["corpus_year", "year"])['piece'])}

        plot_ready_df = plot_ready_df.assign(
            corpus_id=lambda x: x['corpus'].map(corpus_dict),
            piece_id=lambda x: x['piece'].map(piece_dict))

        path_to_plot = "temp_dataframes/CI1_lollipop_pc_content_m2l_df"
        plot_ready_df.to_csv(path_to_plot, sep="\t")

        return path_to_plot, plot_ready_df


class GraphsPrep:
    @staticmethod
    def CI1_scatter_pc_content_index():
        plot_ready_df = pd.read_csv("temp_dataframes/CI1_corpus_pc_content_df", sep='\t')

        for idx, val in enumerate(
                ["m1_GlobalTonic", "m2_GlobalTonic", "m1_LocalTonic", "m2_LocalTonic", "m1_TonicizedTonic",
                 "m2_TonicizedTonic"]):
            fig = px.scatter(plot_ready_df, x="year", y=val, color="corpus",
                             hover_data=['piece', 'global_key', 'ndpc_GlobalTonic', 'local_key', 'ndpc_LocalTonic',
                                         'tonicized_key', 'ndpc_TonicizedTonic'],
                             opacity=0.5, title=f"{val}")
            fig.write_html(f"figures/CI1_scatter_{val}.html")

    @staticmethod
    def CI1_lollipop_pc_content_index_static():
        plot_ready_df = pd.read_csv("temp_dataframes/CI1_lollipop_pc_content_m2l_df", sep='\t')

        # the horizontal lines for corpus mean: ________________________________________________________________________
        df_lines = plot_ready_df.groupby("corpus").agg(start_x=("piece_id", min),
                                                       end_x=("piece_id", max),
                                                       year=("corpus_year", "first"),
                                                       corpus_id=("corpus_id", "first"),
                                                       y=("corpus_avg", "first")).reset_index()

        df_lines = pd.melt(df_lines,
                           id_vars=["corpus", "corpus_id", "year", "y"],
                           value_vars=["start_x", "end_x"],
                           var_name="type",
                           value_name="x")

        df_lines["x_group"] = np.where(df_lines["type"] == "start_x", df_lines["x"] + 0.1, df_lines["x"] - 0.1)
        df_lines["x_group"] = np.where(
            (df_lines["type"] == "start_x").values & (df_lines["x"] == np.min(df_lines["x"])).values,
            df_lines["x_group"] - 0.1,
            df_lines["x_group"]
        )
        df_lines["x_group"] = np.where(
            (df_lines["type"] == "end_x").values & (df_lines["x"] == np.max(df_lines["x"])).values,
            df_lines["x_group"] + 0.1,
            df_lines["x_group"]
        )
        df_lines = df_lines.sort_values(["corpus", "x_group"])

        # the coloring: _______________________________________________________________________________________________
        # Misc colors
        GREY82 = "#d1d1d1"
        GREY70 = "#B3B3B3"
        GREY40 = "#666666"
        GREY30 = "#4d4d4d"
        BG_WHITE = "#fafaf5"

        # These colors (and their dark and light variant) are assigned to each of the 9 seasons
        COLORS = ["#486090", "#D7BFA6", "#04686B", "#d1495b", "#9CCCCC", "#7890A8",
                  "#C7B0C1", "#FFB703", "#B5C9C9", "#90A8C0", "#A8A890", "#ea7317"]

        def adjust_lightness(color, amount=0.5):
            import matplotlib.colors as mc
            import colorsys
            try:
                c = mc.cnames[color]
            except:
                c = color
            c = colorsys.rgb_to_hls(*mc.to_rgb(c))
            return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

        COLORS_DARK = [adjust_lightness(color, 0.8) for color in COLORS]
        COLORS_LIGHT = [adjust_lightness(color, 1.2) for color in COLORS]

        # Three colormaps with three variants
        cmap_regular = mc.LinearSegmentedColormap.from_list("regular", COLORS)
        cmap_dark = mc.LinearSegmentedColormap.from_list("dark", COLORS_DARK)
        cmap_light = mc.LinearSegmentedColormap.from_list("light", COLORS_LIGHT)

        # Function used to normalize corpus_id values into 0-1 scale.
        normalize = mc.Normalize(vmin=1, vmax=plot_ready_df["corpus_id"].max())

        # Horizontal lines
        HLINES = [-40, -20, 0, 20, 40, 60, 80]

        # lollipop dot size: __________________________________________________________________________________________
        CHORDS_MAX = plot_ready_df["chord_num"].max()
        CHORDS_MIN = plot_ready_df["chord_num"].min()

        # low and high refer to the final dot size.
        def scale_to_interval(x, low=15, high=150):
            return ((x - CHORDS_MIN) / (CHORDS_MAX - CHORDS_MIN)) * (high - low) + low

        # fig starts : __________________________________________________________________________________________
        fig, ax = plt.subplots(figsize=(35, 20))

        # Some layout stuff ----------------------------------------------
        # Background color
        fig.patch.set_facecolor(BG_WHITE)
        ax.set_facecolor(BG_WHITE)

        # First, horizontal lines that are used as scale reference
        # zorder=0 to keep them in the background
        for h in HLINES:
            plt.axhline(h, color=GREY82, zorder=0)

        # Add vertical segments ------------------------------------------
        # Vertical segments.
        # These represent the deviation of piece's ci value from the mean ci value of the corpus they belong.
        plt.vlines(
            x="piece_id",
            ymin="min_val",
            ymax="max_val",
            color=cmap_light(normalize(plot_ready_df["corpus_id"])),
            data=plot_ready_df
        )

        # Add horizontal segments ----------------------------------------
        # A grey line that connects mean values
        # The third argument is the format string, either empty or "-"
        # plt.plot("x", "y", "-", color=GREY40, data=df_lines)

        # These represent the mean corpus stats.
        for corpus in df_lines["corpus"].unique():
            d = df_lines[df_lines["corpus"] == corpus]
            plt.plot("x_group", "y", "",
                     color=cmap_dark(normalize(df_lines[df_lines["corpus"] == corpus]["corpus_id"].values[0])),
                     lw=5,
                     data=d, solid_capstyle="butt")

        # Add dots -------------------------------------------------------
        # The dots indicate each piece's value, with its size given by the number of chords in the piece.
        plt.scatter(
            "piece_id",
            "max_val",
            s=scale_to_interval(plot_ready_df["chord_num"]),
            color=cmap_regular(normalize(plot_ready_df["corpus_id"])),
            data=plot_ready_df,
            zorder=3
        )
        plt.scatter(
            "piece_id",
            "min_val",
            s=scale_to_interval(plot_ready_df["chord_num"]),
            color=cmap_regular(normalize(plot_ready_df["corpus_id"])),
            data=plot_ready_df,
            zorder=3
        )

        # Add labels -----------------------------------------------------
        # They indicate the corpus and free us from using a legend.

        corpus_label_midpoint = df_lines.groupby("corpus")["x"].mean()
        corpus_label_pos_tuple_list = [(corpus, avg_x) for corpus, avg_x in
                                       zip(corpus_label_midpoint.index, corpus_label_midpoint)]

        for corpus, midpoint in corpus_label_pos_tuple_list:
            color = cmap_dark(normalize(df_lines[df_lines["corpus"] == corpus]["corpus_id"].values[0] + 1))
            plt.text(
                midpoint, 85, f"{corpus}",
                color=color,
                weight="bold",
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1,
                    boxstyle="round",
                    pad=0.2
                )
            )

        # Customize layout -----------------------------------------------

        # Hide spines
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["left"].set_color("none")

        # Customize y ticks
        # * Remove y axis ticks
        # * Put labels on both right and left sides
        plt.tick_params(axis="y", labelright=True, length=0)
        plt.yticks(HLINES, fontsize=11, color=GREY30)
        plt.ylim(0.98 * (-40), 80 * 1.02)

        # Remove ticks and legends
        plt.xticks([], "")

        # Y label
        plt.ylabel("Pitch class content index (mean)", fontsize=14)
        # plt.xlabel("Time (year)", fontsize=14)

        # Save the plot!
        plt.savefig(
            "figures/CI1_lollipop_m2l.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3
        )



# TEST ===============================================================

def test():
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')

    df_row = df.iloc[62]
    numeral = Numeral.from_df(df_row)

    result = ...
    print(f'{numeral=}')
    print(f'{result=}')


def test2():
    tchaikovsky = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/tchaikovsky_seasons/harmonies/op37a12.tsv',
        sep='\t')

    debussy = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')

    result = piece_wise_operation(piece_df=debussy, chord_wise_operation=CI2_multilevel_ci_dict)
    print(f'{result=}')


if __name__ == '__main__':
    GraphsPrep.CI1_lollipop_pc_content_index_static()
