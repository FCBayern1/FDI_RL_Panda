import pandapower as pp
import pandas as pd
import numpy as np
from pandapower.control import ConstControl
from pandapower.timeseries import run_timeseries, OutputWriter
import utils.network as network

# Step 1: Setup Network and Load Profile
time_steps = 200
net = network.create_network()

load_profile_df = network.create_load_profile(time_steps, base_load=50, load_amplitude=30)
ds = pp.timeseries.DFData(load_profile_df)

# Step 2: Attach `ConstControl` for Load Variation
for load_idx in net.load.index:
    ConstControl(net, element="load", variable="p_mw", element_index=[load_idx], data_source=ds, profile_name="p_mw")

# Step 3: Configure Output Logging
output_path = "./output_load_experiment"
log_vars = [("load", "p_mw"), ("trafo", "loading_percent"), ("bus", "vm_pu")]

ow = OutputWriter(net, time_steps=range(time_steps), output_path=output_path, log_variables=log_vars, output_file_type=".csv")

# Step 4: Define a Function to Monitor Transformer Overload
def monitor_overload(net, time_step, pf_converged, ctrl_converged, ts_variables, **kwargs):
    """ Logs transformer overloads and bus voltage issues at each time step. """

    # Transformer Overload Detection
    overloaded_trafos = net.res_trafo[net.res_trafo["loading_percent"] > 100]
    if not overloaded_trafos.empty:
        print(f"⚠️ Time Step {time_step}: Transformer Overload Detected!")
        print(overloaded_trafos)

    # Low Voltage Detection
    low_voltage_buses = net.res_bus[net.res_bus["vm_pu"] < 0.9]
    if not low_voltage_buses.empty:
        print(f"⚠️ Time Step {time_step}: Low Voltage Buses Detected!")
        print(low_voltage_buses)

# Step 5: Run Time-Series Simulation with Error Handling
try:
    run_timeseries(net, time_steps=range(time_steps), output_writer_fct=monitor_overload)
except pp.powerflow.LoadflowNotConverged:
    print("⚠️ Simulation stopped due to LoadflowNotConverged. Consider adjusting load variations.")

# Step 6: Load and Display Results
load_results = pd.read_csv(f"{output_path}/load/p_mw.csv", sep=";")
trafo_results = pd.read_csv(f"{output_path}/trafo/loading_percent.csv", sep=";")
bus_results = pd.read_csv(f"{output_path}/bus/vm_pu.csv", sep=";")

print("\n🔍 Load Changes Over Time:")
print(load_results.head(10))

print("\n🔍 Transformer Loading Over Time:")
print(trafo_results.head(10))

print("\n🔍 Bus Voltage Over Time:")
print(bus_results.head(10))
