import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.ticker as ticker

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--parteo-front-filename",
                        type=str,
                        default="pareto-front.json")
    parser.add_argument("--x-axis-name", type=str, default="click_loss")
    parser.add_argument("--y-axis-name", type=str, default="order_loss")
    parser.add_argument("--title", type=str, default="Clicks vs. Orders")
    parser.add_argument("--output-filename",
                        type=str,
                        default="pareto-front.png")
    parser.add_argument("--plot", type=bool, default=False)

    args = parser.parse_args()

    with open(args.parteo_front_filename, "r") as f:
        data = json.loads(f.readline())

    x_name = args.x_axis_name.replace("_", " ")
    y_name = args.y_axis_name.replace("_", " ")
    df = pd.DataFrame(zip(data[args.x_axis_name], data[args.y_axis_name],
                          data["order_weight"]),
                      columns=[x_name, y_name, "order weight"])

    cmap = plt.get_cmap("cool")
    fig, axes = plt.subplots(1, 1, layout='constrained')

    min_clicks = min(df["click loss"])
    max_clicks = max(df["click loss"])
    min_orders = min(df["order loss"])
    max_orders = max(df["order loss"])

    sc = axes.scatter(x=df["click loss"],
                      y=df["order loss"],
                      c=cmap(df["order weight"]))
    axes.plot(df["click loss"], df["order loss"], c="grey")
    axes.title.set_text("Diginetica")
    axes.set_xlabel("$\mathcal{L}_c$")
    axes.set_ylabel("$\mathcal{L}_o$")
    axes.set_xticks(
        np.arange(min_clicks, max_clicks + (max_clicks - min_clicks) / 2,
                  (max_clicks - min_clicks) / 2))
    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    axes.set_yticks(
        np.arange(min_orders, max_orders + (max_orders - min_orders) / 2,
                  (max_orders - min_orders) / 2))
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

    sm = ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right")
    cbar.ax.set_title("$\pi_o$")

    plt.savefig(args.output_filename)

    if args.plot:
        plt.show()
