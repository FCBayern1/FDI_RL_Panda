import os

import pandapower as pp
import pandas as pd
import numpy as np
from pandapower.control import ConstControl
from pandapower.timeseries import run_timeseries, OutputWriter
import utils.network as network
from controllers.MonitorController import MonitorController
from plots.plot_utils import plot_curves

time_steps = 200
net = network.create_network()

load_profile_df = network.create_load_profile(time_steps, base_load=20, load_amplitude=20)
ds = pp.timeseries.DFData(load_profile_df)

for load_idx in net.load.index:
    print(load_idx)
    ConstControl(net, element="load", variable="p_mw", element_index=[load_idx], data_source=ds, profile_name="p_mw")

monitor = MonitorController(net)

output_path = "./output_load_experiment"

os.makedirs(output_path, exist_ok=True)

log_vars = [
    ("res_load", "p_mw"),
    ("res_trafo", "loading_percent"),
    ("res_bus", "vm_pu")
]

ow = OutputWriter(
    net, time_steps=time_steps, output_path=output_path, log_variables=log_vars, output_file_type=".csv", csv_separator=';'
)

try:
    run_timeseries(net, time_steps=range(time_steps))
except pp.powerflow.LoadflowNotConverged:
    print("⚠️ Simulation stopped due to Loadflow NotConverged. Consider adjusting load variations.")


try:
    load_results = pd.read_csv(f"{output_path}/res_load/p_mw.csv", sep=";")
    trafo_results = pd.read_csv(f"{output_path}/res_trafo/loading_percent.csv", sep=";")
    bus_results = pd.read_csv(f"{output_path}/res_bus/vm_pu.csv", sep=";")

    print("\n🔍 Transformer Loading Over Time:")
    print(trafo_results.head(10))

except FileNotFoundError as e:
    print("File not found. Please ensure the OutputWriter has logged the correct data.")

plot_curves(f"{output_path}/res_trafo/loading_percent.csv")