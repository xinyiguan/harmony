import json
import urllib
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px




def scatter_plot_historical_CI1_pc_content_index():
    plot_ready_df = PlotCI1_PitchClassContentIndex.load_data()

    for idx, val in enumerate(["m1_GlobalTonic", "m2_GlobalTonic", "m1_LocalTonic", "m2_LocalTonic", "m1_TonicizedTonic", "m2_TonicizedTonic"]):
        fig = px.scatter(plot_ready_df, x="year", y=val, color="corpus",
                           hover_data=['piece', 'global_key', 'ndpc_GlobalTonic', 'local_key', 'ndpc_LocalTonic',
                                       'tonicized_key', 'ndpc_TonicizedTonic'],
                           opacity=0.5, title=f"{val}")
        fig.write_html(f"figures/{val}.html")


    fig_m2_lt = px.scatter(plot_ready_df, x="year", y="m2_LocalTonic", color="corpus",
                           hover_data=['piece', 'global_key', 'ndpc_GlobalTonic', 'local_key', 'ndpc_LocalTonic',
                                       'tonicized_key', 'ndpc_TonicizedTonic'],
                           opacity=0.5, title="ref key: Local key, d5th ref: chord root")
    fig_m2_lt.write_html("figures/fig_m2_lt.html")

    fig_m1_lt = px.scatter(plot_ready_df, x="year", y="m1_LocalTonic", color="corpus",
                           hover_data=['piece', 'global_key', 'ndpc_GlobalTonic', 'local_key', 'ndpc_LocalTonic',
                                       'tonicized_key', 'ndpc_TonicizedTonic'],
                           opacity=0.5, title="ref key: Local key, d5th ref: C")
    fig_m1_lt.write_html("figures/fig_m1_lt.html")

    fig_m2_tt = px.scatter(plot_ready_df, x="year", y="m2_TonicizedTonic", color="corpus",
                           hover_data=['piece', 'global_key', 'ndpc_GlobalTonic', 'local_key', 'ndpc_LocalTonic',
                                       'tonicized_key', 'ndpc_TonicizedTonic'],
                           opacity=0.5, title="ref key: Chord root as tonic, d5th ref: chord root")
    fig_m2_tt.write_html("figures/fig_m2_tt.html")

    fig_m1_tt = px.scatter(plot_ready_df, x="year", y="m1_TonicizedTonic", color="corpus",
                           hover_data=['piece', 'global_key', 'ndpc_GlobalTonic', 'local_key', 'ndpc_LocalTonic',
                                       'tonicized_key', 'ndpc_TonicizedTonic'],
                           opacity=0.5, title="ref key: Chord root as tonic, d5th ref: C")
    fig_m1_tt.write_html("figures/fig_m1_tt.html")

    fig_m2_gt = px.scatter(plot_ready_df, x="year", y="m2_GlobalTonic", color="corpus",
                           hover_data=['piece', 'global_key', 'ndpc_GlobalTonic', 'local_key', 'ndpc_LocalTonic',
                                       'tonicized_key', 'ndpc_TonicizedTonic'],
                           opacity=0.5, title="ref key: Global key, d5th ref: chord root")
    fig_m2_gt.write_html("figures/fig_m2_gt.html")

    fig_m1_gt = px.scatter(plot_ready_df, x="year", y="m1_GlobalTonic", color="corpus",
                           hover_data=['piece', 'global_key', 'ndpc_GlobalTonic', 'local_key', 'ndpc_LocalTonic',
                                       'tonicized_key', 'ndpc_TonicizedTonic'],
                           opacity=0.5, title="ref key: Global key, d5th ref: C")
    fig_m1_gt.write_html("figures/fig_m1_gt.html")





if __name__ == "__main__":
    # url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
    # with urlopen(url) as response:
    #     body = response.read()
    # json_data = json.loads(body)
    #
    # python_data = json.loads(json_data)
    # df = pd.json_normalize(python_data)
    # print(df.head())

    import matplotlib.pyplot as plt

    # Define the list of colors
    colors = ["#486090", "#D7BFA6", "#04686B", "#d1495b", "#9CCCCC", "#7890A8",
              "#C7B0C1", "#FFB703", "#B5C9C9", "#90A8C0", "#A8A890", "#ea7317"]

    # Plot a bar chart with each color as a bar
    fig, ax = plt.subplots()
    for i in range(len(colors)):
        ax.bar(i, 1, color=colors[i], edgecolor='black')

    # Set the x-tick labels and axis limits
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels(colors, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(colors) - 0.5)

    # Show the plot
    plt.show()
