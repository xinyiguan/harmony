import json
import urllib
from typing import Literal
from urllib.request import urlopen
import matplotlib.colors as mc
from plotly.offline import init_notebook_mode, iplot

import numpy as np
import pandas as pd
import plotly.express as px

import plotly.graph_objs as go


def CI1_lollipop_pc_content_index():
    plot_ready_df = pd.read_csv("temp_dataframes/CI1_lollipop_pc_content_m2l_df", sep='\t')

    pretty_corpus_labels = {"corelli": "Corelli",
                            "mozart_piano_sonatas": "Mozart Sonata",
                            "beethoven_piano_sonatas": "Beethoven Sonata",
                            "ABC": "ABC",
                            "chopin_mazurkas": "Chopin Mazurka",
                            "schumann_kinderszenen": "Schumann Kinderszenen",
                            "liszt_pelerinage": "Liszt",
                            "tchaikovsky_seasons": "Tchaikovsky",
                            "dvorak_silhouettes": "Dvorak",
                            "grieg_lyric_pieces": "Grieg",
                            "debussy_suite_bergamasque": "Debussy",
                            "medtner_tales": "Medtner"
                            }

    # the horizontal lines for corpus mean: ________________________________________________________________________
    df_lines = plot_ready_df.groupby("corpus").agg(start_x=("piece_id", min),
                                                   end_x=("piece_id", max),
                                                   year=("corpus_year", "first"),
                                                   corpus_id=("corpus_id", "first"),
                                                   y=("corpus_avg", "first")).sort_values(["year"]).reset_index()

    # add some padding (epsilon) around the corpus line group
    df_lines["start_x_eps"] = df_lines["start_x"] - 0.1
    df_lines["end_x_eps"] = df_lines["end_x"] + 0.1

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

    # Add annotation of the data: def of metrics as the title
    set_M_annotation_dict = {"M1": r'$M_1 = \{d_{5th} (rt=C, a=nd(k=GlobalKey, A))}$',
                             "M2": r'$M_2 = \{d_{5th} (rt=root, a=nd(k=GlobalKey, A))}$',
                             "M3": r'$M_3 = \{d_{5th} (rt=C, a=nd(k=LocalKey, A))}$',
                             "M4": r'$M_4 = \{d_{5th} (rt=root, a=nd(k=LocalKey, A))}$',
                             "M5": r'$M_5 = \{d_{5th} (rt=C, a=nd(k=Tonicization(r, A), A))}$',
                             "M6": r'$M_6 = \{d_{5th} (rt=root, a=nd(k=Tonicization(r,A), A))}$',
                             }

    # Customize layout -----------------------------------------------
    layout = go.Layout(title=set_M_annotation_dict["M4"],
                       # xaxis=go.layout.XAxis(
                       #     title='Pieces (* almost in chronological order)',
                       #     showticklabels=False),
                       # yaxis=go.layout.YAxis(
                       #     title='Values'
                       # ),
                       showlegend=False)

    # Fig starts : __________________________________________________________________________________________
    fig = go.Figure(layout=layout)

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

    # loop through each unique corpus in the data and plot a horizontal line
    for corpus in df_lines["corpus"].unique():
        d = df_lines[df_lines["corpus"] == corpus]
        fig.add_shape(type="line",
                      x0=d["start_x_eps"].values[0],
                      y0=d["y"].values[0],
                      x1=d["end_x_eps"].values[0],
                      y1=d["y"].values[0],
                      line=dict(color=corpus_color_dict[d["corpus"].values[0]], width=3))

    # Add dots -------------------------------------------------------
    # The dots indicate each piece's value, with its size given by the number of chords in the piece.

    # hacking the color for each corpus: add a color column to the df
    plot_ready_df['colors'] = plot_ready_df['corpus'].apply(lambda x: corpus_color_dict[x])

    fig.add_scatter(x=plot_ready_df["piece_id"],
                    y=plot_ready_df["min_val"],
                    mode="markers",
                    marker=dict(size=scale_to_interval(plot_ready_df["chord_num"]),
                                color=plot_ready_df.colors),
                    customdata=np.stack((plot_ready_df['piece'], plot_ready_df['year'], plot_ready_df['chord_num'],
                                         plot_ready_df['min_val']), axis=-1),
                    hovertemplate='<b>Piece</b>: %{customdata[0]}<br>' +
                                  '<b>Year</b>: %{customdata[1]}<br>' +
                                  '<b>Chord num</b>: %{customdata[2]}<br>' +
                                  '<b>Value</b>: %{customdata[3]}' +
                                  '<extra></extra>')

    fig.add_scatter(x=plot_ready_df["piece_id"],
                    y=plot_ready_df["max_val"],
                    mode="markers",
                    marker=dict(size=scale_to_interval(plot_ready_df["chord_num"]),
                                color=plot_ready_df.colors),
                    customdata=np.stack((plot_ready_df['piece'], plot_ready_df['year'], plot_ready_df['chord_num'],
                                         plot_ready_df['max_val']), axis=-1),
                    hovertemplate='<b>Piece</b>: %{customdata[0]}<br>' +
                                  '<b>Year</b>: %{customdata[1]}<br>' +
                                  '<b>Chord num</b>: %{customdata[2]}<br>' +
                                  '<b>Value</b>: %{customdata[3]}' +
                                  '<extra></extra>')

    # Add labels -----------------------------------------------------
    # They indicate the corpus and free us from using a legend.

    df_lines["corpus_label_x"] = df_lines.apply(lambda row: (row["start_x"] + row["end_x"]) / 2, axis=1)
    # to avoid overlapping of labels, create alternating heights
    df_lines["corpus_label_y"] = [max(HLINES) + x for x in
                                  [0 if i % 3 == 0 else 5 if i % 3 == 1 else -5 for i in range(df_lines.shape[0])]]

    corpus_label_pos_tuple_list = [(corpus, x_pos, y_pos) for corpus, x_pos, y_pos in
                                   zip(df_lines["corpus"], df_lines["corpus_label_x"], df_lines["corpus_label_y"])]

    for corpus, x_pos, y_pos in corpus_label_pos_tuple_list:
        color = corpus_color_dict[corpus]
        fig.add_annotation(
            x=x_pos, y=y_pos, text=f"{corpus}",
            showarrow=False,
            font=dict(
                # size=16,
                color=color
            ),
            align="center",
            bordercolor=color,
            borderwidth=1
        )

    # fig.add_annotation(x=max(plot_ready_df["piece_id"]) + 50,
    #                    y=min(HLINES) + 20,
    #                    text='<b>Definition</b> of set $M$:<br>' +
    #                         '%{set_M_annotation_dict["M4"]} <br>',
    #                    showarrow=False,
    #                    bordercolor=GREY70,
    #                    borderwidth=1
    #                    )

    # Add custom legend for bubble size ----------------------------------------------
    # Horizontal position for the dots and their labels
    def generate_dot_legend_pos(middle_number: int, interval: int):
        return [middle_number - (interval * 2), middle_number - interval, middle_number + interval,
                middle_number + (interval * 2)]

    x_pos = generate_dot_legend_pos(middle_number=plot_ready_df["piece_id"].mean(), interval=4)
    chord_num = np.array([150, 300, 450, 600])

    fig.add_scatter(x=x_pos,
                    y=[min(plot_ready_df["min_val"]) - 20] * len(x_pos),
                    mode="markers",
                    marker=dict(size=scale_to_interval(chord_num), color="black")

                    )
    fig.add_annotation(x=np.mean(x_pos),
                       y=min(plot_ready_df["min_val"]) - 30,
                       text="Number of chords per piece",
                       showarrow=False)

    fig.update_layout(showlegend=False,
                      xaxis_title="Pieces (* almost in chronological order)",
                      yaxis_title="Values")
    fig.write_html(f"figures/CI1_lollipop.html", include_plotlyjs='cdn', full_html=True)


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
