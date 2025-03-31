import random

import pandas as pd
import torch
from models.dqn_agent import DQNAgent
import numpy as np

from pandapower.control import ConstControl
import pandapower as pp
from pandapower.timeseries import run_timeseries, OutputWriter
from controllers.rl_transformer_control import DQNTransformerController
from envs.multi_agent_substation_env import multi_agent_substation_env
from controllers.transformer_control import TransformerDisconnect
from plots.plot_utils import plot_curves, plot_confusion_matrix
from utils.Generate_fdi import generate_fdi_list
from utils.network import create_network, create_load_profile, create_stable_gen_profile

time_steps = 100

net = create_network()

trafo_indices = list(net.trafo.index)

# initialise the transformer temperature
for i in trafo_indices:
    net.trafo.at[i,"temperature_measured"] = 20.0

print(net.trafo.columns)

overload_steps = [20, 50, 80]

load_profile_df = create_load_profile(time_steps, base_load=2, load_amplitude=4, overload_steps=overload_steps, overload_factor=3.0)
ds = pp.timeseries.DFData(load_profile_df)

for load_idx in net.load.index:
    ConstControl(net, element="load", variable="p_mw", element_index=[load_idx], data_source=ds, profile_name="p_mw")

T_ambient = 25.0
T_rated = 65.0
n = 1.6
max_temperature = 100.0

# # Re-attach the generator control
gen_profile = create_stable_gen_profile(net, time_steps=time_steps, base_gen_factor=2.0)
gen_ds = pp.timeseries.DFData(gen_profile)
control = ConstControl(net, element='gen', variable='p_mw', element_index=[0], data_source=gen_profile, profile_name='p_mw')

# Generate FDI lists for each transformer
num_attacks = 20
min_faulty_data = 105.0
max_faulty_data = 110.0

fdi_list = generate_fdi_list(time_steps, num_attacks, min_faulty_data, max_faulty_data)

fdi_per_trafo = [[] for _ in trafo_indices]
fdi_attack_log = {}  # Log to track FDI attacks for later analysis

# Distribute FDI attacks across transformers
for fdi in fdi_list:
    target_trafo_index = random.choice(range(len(trafo_indices)))
    fdi_per_trafo[target_trafo_index].append(fdi)
    # Log each FDI attack: {time_step: (trafo_index, faulty_temperature)}
    time_step, faulty_temperature = fdi
    fdi_attack_log[(time_step, target_trafo_index)] = faulty_temperature

# Print the FDI attack log to review all attacks before training
for (time_step, trafo_index), faulty_temperature in sorted(fdi_attack_log.items()):
    print(f"Time step {time_step}, Transformer {trafo_index}: Faulty temperature = {faulty_temperature}°C")

# Add the Transformer Protection Controllers with specific FDI lists
for i, index in enumerate(trafo_indices):
    trafo_protection = TransformerDisconnect(
        net=net,
        trafo_index=index,
        max_temperature=max_temperature,      # Set the maximum allowable temperature
        T_ambient=T_ambient,                  # Ambient temperature
        T_rated=T_rated,                    # Rated temperature rise
        n=n,                                  # Exponent for temperature calculation
        fdi_list=fdi_per_trafo[i],            # Apply specific FDI list for each transformer
        total_steps=time_steps
    )

env = multi_agent_substation_env(net,
                trafo_indices=trafo_indices,
                delta_p=0.05,
                initial_p=0.0,
                voltage_tolerance=0.05,
                voltage_penalty_factor=10.0,
                line_loading_limit=100.0,
                power_flow_penalty_factor=5.0,
                load_reward_factor=20.0,
                transformer_reward_factor=20.0,
                disconnection_penalty_factor=50.0,
                total_steps=time_steps)

env.reset()

for i, index in enumerate(trafo_indices):
    model_path = f"trafo_{index}_dqn_model.pth"
    trafo_controller = DQNTransformerController(net,
                env,
                trafo_index=index,
                max_temperature=max_temperature,
                # Set the maximum allowable temperature
                T_ambient=T_ambient,  # Ambient temperature
                T_rated=T_rated,  # Rated temperature rise
                n=n,  # Exponent for temperature calculation
                fdi_list=fdi_per_trafo[i],
                total_steps=time_steps,
                model_path = model_path)

log_vars = [
    ("trafo", "in_service"),  # Record transformer status
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured")
]
output_path = "./results"
ow = OutputWriter(
    net, time_steps=time_steps, output_path= output_path,
    output_file_type='.csv', log_variables=log_vars, csv_separator=';'
)

# print(net.controller)
# print("before")

run_timeseries(net, time_steps = range(time_steps))

# print(net.controller)
# print("after")

try:
    transformer_status_log = pd.read_csv(f'{output_path}/trafo/in_service.csv', sep=';')
    transformer_loading = pd.read_csv(f'{output_path}/res_trafo/loading_percent.csv', sep=';')
    # Clean up the data
    transformer_status_log.drop(columns=['Unnamed: 0'], inplace=True)

except FileNotFoundError as e:
    print("File not found. Please ensure the OutputWriter has logged the correct data.")

plot_curves(f"{output_path}/res_trafo/loading_percent.csv", f"{output_path}/loading_curves.png")

# TP, FP, TN, FN
dqn_controllers = [ctrl for ctrl in net.controller["object"] if isinstance(ctrl, DQNTransformerController)]

total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

print("**Transformer-Level Statistics**")
for controller in dqn_controllers:
    print(f"Transformer {controller.trafo_index}: TP={controller.tp}, FP={controller.fp}, TN={controller.tn}, FN={controller.fn}")
    total_tp += controller.tp
    total_fp += controller.fp
    total_tn += controller.tn
    total_fn += controller.fn

print(f"✅ True Positive (TP) = {total_tp}")
print(f"❌ False Positive (FP) = {total_fp}")
print(f"✅ True Negative (TN) = {total_tn}")
print(f"❌ False Negative (FN) = {total_fn}")
conf_matrix = np.array([[total_tp, total_fp], [total_fn, total_tn]])
plot_confusion_matrix(conf_matrix)




