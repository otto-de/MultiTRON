import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.ticker as ticker

def infer_weight(data):
    if "order_weight" in data.keys(): 
        return "order"
    else: return "distortion"

def infer_label(label):
    if label == "click loss":
        return "$\mathcal{L}_c$"
    elif label == "order loss":
        return "$\mathcal{L}_o$"
    elif label == "distortion loss":
        return "$\mathcal{L}_d$"
    else: return label

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

    loss_type = infer_weight(data)

    x_name = args.x_axis_name.replace("_", " ")
    y_name = args.y_axis_name.replace("_", " ")
    df = pd.DataFrame(zip(data[args.x_axis_name], data[args.y_axis_name],
                          data[f"{loss_type}_weight"]),
                      columns=[x_name, y_name, f"{loss_type} weight"])


    cmap = plt.get_cmap("cool")
    fig, axes = plt.subplots(1, 1, layout='constrained')

    min_clicks = min(df[x_name])
    max_clicks = max(df[x_name])
    min_second = min(df[y_name])
    max_second = max(df[y_name])

    sc = axes.scatter(x=df[x_name],
                      y=df[y_name],
                      c=cmap(df[f"{loss_type} weight"]))
    axes.plot(df[x_name], df[y_name], c="grey")
    axes.title.set_text(args.title)

    axes.set_xlabel(infer_label(x_name))
    axes.set_ylabel(infer_label(y_name))
    axes.set_xticks(
        np.arange(min_clicks, max_clicks + (max_clicks - min_clicks) / 2,
                  (max_clicks - min_clicks) / 2))
    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    axes.set_yticks(
        np.arange(min_second, max_second + (max_second - min_second) / 2,
                  (max_second - min_second) / 2))
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

    sm = ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right")

    cbar.ax.set_title("$\pi_o$" if loss_type=="order" else "$\pi_d$")

    plt.savefig(args.output_filename)

    if args.plot:
        plt.show()
