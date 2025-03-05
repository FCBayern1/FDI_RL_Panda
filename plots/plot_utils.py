import pandas as pd
from pandapower.plotting.plotly import vlevel_plotly
from utils.network import create_network
import os
import matplotlib.pyplot as plt
import plotly.io as pio

def plot_network():
    fig = vlevel_plotly(create_network())
    fig.write_image("pics/voltage_level_plot.png")

def plot_curves(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found!!!!!!")
    trafo_results = pd.read_csv(file_path, sep = ";")
    time_steps = trafo_results.index

    plt.figure(figsize = (10, 5))
    for trafo_id in trafo_results.columns[1:]:
        plt.plot(time_steps, trafo_results[trafo_id]*100, label = f"Transformer {trafo_id}")
    plt.xlabel("Time Step")
    plt.ylabel("Transformer Loading (%)")
    plt.title("Transformer Loading Over Time")

    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig("/Users/joshua/PandaPower/plots/pics/curves.png")

