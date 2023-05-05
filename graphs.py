import json
import urllib
from typing import Literal
from urllib.request import urlopen
import matplotlib.colors as mc

import numpy as np
import pandas as pd
import plotly.express as px

import plotly.graph_objs as go


def CI1_lollipop_pc_content_index():
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

    # These colors (and their dark and light variant) are assigned to each of the 12 corpora
    COLORS = ["#486090", "#D7BFA6", "#04686B", "#d1495b", "#9CCCCC", "#7890A8",
              "#C7B0C1", "#FFB703", "#B5C9C9", "#90A8C0", "#A8A890", "#ea7317"]

    # Coloring the pieces in each corpus, each corpus corresponds to one color
    corpus_color_dict = {corpus: color for corpus, color in zip(plot_ready_df['corpus'].unique(), COLORS)}

    # Horizontal lines
    HLINES = [-40, -20, 0, 20, 40, 60, 80]

    # lollipop dot size: __________________________________________________________________________________________
    CHORDS_MAX = plot_ready_df["chord_num"].max()
    CHORDS_MIN = plot_ready_df["chord_num"].min()

    # low and high refer to the final dot size.
    def scale_to_interval(x, low=5, high=30):
        return ((x - CHORDS_MIN) / (CHORDS_MAX - CHORDS_MIN)) * (high - low) + low

    # fig starts : __________________________________________________________________________________________
    fig = go.Figure()

    # Some layout stuff ----------------------------------------------
    # Background color

    fig.update_layout(
        paper_bgcolor=BG_WHITE,
        plot_bgcolor=BG_WHITE)

    # Add horizontal lines that are used as scale reference
    # layer="below" to keep them in the background

    for h in HLINES:
        fig.add_hline(y=h, layer="below", line_width=1, line_color=GREY40)

    # Add vertical segments ------------------------------------------
    # Vertical segments.
    # These represent the deviation of piece's ci value from the mean ci value of the corpus they belong.

    for piece in plot_ready_df["piece"]:
        p = plot_ready_df[plot_ready_df["piece"] == piece]
        fig.add_shape(
            type="line",
            x0=p["piece_id"].values[0],
            x1=p["piece_id"].values[0],
            y0=p["min_val"].values[0],
            y1=p["max_val"].values[0],
            line=dict(color=corpus_color_dict[p["corpus"].values[0]], width=1)
        )

    # Add horizontal segments ----------------------------------------
    # These represent the mean corpus stats.

    for corpus in df_lines["corpus"].unique():
        d = df_lines[df_lines["corpus"] == corpus]
        for i, row in d.iterrows():
            # if the type is start_x, then set x0 to the corresponding x value
            if row['type'] == 'start_x':
                x0 = row['x']
            # if the type is end_x, then set x1 to the corresponding x value
            elif row['type'] == 'end_x':
                x1 = row['x']
                # add a new shape object to the figure with the corresponding x0, x1, and y values
                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=row['y'],
                    x1=x1,
                    y1=row['y'],
                    line=dict(color=corpus_color_dict[d["corpus"].values[0]], width=3)
                )

    # Add dots -------------------------------------------------------
    # The dots indicate each piece's value, with its size given by the number of chords in the piece.

    fig.add_scatter(x=plot_ready_df["piece_id"],
                    y=plot_ready_df["min_val"],
                    mode="markers",
                    marker=dict(size=scale_to_interval(plot_ready_df["chord_num"]),
                                color=corpus_color_dict[plot_ready_df["corpus"].values[0]]
                                ),
                    text=plot_ready_df["corpus", "piece", "year", "chord_num", "min_val"]
                    )

    fig.add_scatter(x=plot_ready_df["piece_id"],
                    y=plot_ready_df["max_val"],
                    mode="markers",
                    marker=dict(size=scale_to_interval(plot_ready_df["chord_num"]),
                                color=corpus_color_dict[plot_ready_df["corpus"].values[0]]
                                ),
                    text=plot_ready_df["corpus", "piece", "year", "chord_num", "max_val"]
                    )







    fig.write_html(f"figures/CI1_lollipop.html")
    print(df)
    assert False

    # Add labels -----------------------------------------------------
    # They indicate the corpus and free us from using a legend.

    corpus_label_midpoint = df_lines.groupby("corpus")["x"].mean()
    corpus_label_pos_tuple_list = [(corpus, avg_x) for corpus, avg_x in
                                   zip(corpus_label_midpoint.index, corpus_label_midpoint)]

    for corpus, midpoint in corpus_label_pos_tuple_list:
        color = cmap_dark(normalize(df_lines[df_lines["corpus"] == corpus]["corpus_id"].values[0] + 1))
        # plt.text(
        #     midpoint, 85, f"{corpus}",
        #     color=color,
        #     weight="bold",
        #     ha="center",
        #     va="center",
        #     fontsize=10,
        #     bbox=dict(
        #         facecolor="none",
        #         edgecolor=color,
        #         linewidth=1,
        #         boxstyle="round",
        #         pad=0.2
        #     )
        # )

        fig.add_annotation(
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


def test_color():
    def adjust_lightness(color, amount=0.5):
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    import matplotlib.pyplot as plt
    import matplotlib.colors as mc

    # Define the list of colors
    COLORS = ["#486090", "#D7BFA6", "#04686B", "#d1495b", "#9CCCCC", "#7890A8",
              "#C7B0C1", "#FFB703", "#B5C9C9", "#90A8C0", "#A8A890", "#ea7317"]

    COLORS_DARK = [adjust_lightness(color, 0.8) for color in COLORS]
    COLORS_LIGHT = [adjust_lightness(color, 1.2) for color in COLORS]

    # Three colormaps with three variants
    cmap_regular = mc.LinearSegmentedColormap.from_list("regular", COLORS)
    cmap_dark = mc.LinearSegmentedColormap.from_list("dark", COLORS_DARK)
    cmap_light = mc.LinearSegmentedColormap.from_list("light", COLORS_LIGHT)

    # Plot a bar chart with each color as a bar
    fig, ax = plt.subplots()
    for i in range(len(COLORS)):
        ax.bar(i, 1, color=COLORS[i], edgecolor='black')

    # Set the x-tick labels and axis limits
    ax.set_xticks(range(len(COLORS)))
    ax.set_xticklabels(COLORS, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(COLORS) - 0.5)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    CI1_lollipop_pc_content_index()
