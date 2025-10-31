xuanimport random

import pandas as pd
import torch
from src.networks.dqn_agent import DQNAgent
import numpy as np

from pandapower.control import ConstControl
import pandapower as pp
from pandapower.timeseries import run_timeseries, OutputWriter, DFData
from src.controllers.rl_transformer_control import DQNTransformerController
from src.envs.multi_agent_substation_env import multi_agent_substation_env
from src.controllers.transformer_control import TransformerDisconnect
from src.plots.plot_utils import plot_curves, plot_confusion_matrix
from src.utils.Generate_fdi import generate_fdi_list
from src.utils.network import create_network, create_load_profile, create_stable_gen_profile, create_30_network

output_path = "./dqn/output_data_test"
time_steps = 200
T_ambient = 25.0
T_rated = 65.0
n = 1.6
max_temperature = 147.44
# Generate FDI lists for each transformer
num_attacks = 20
min_faulty_data = 160.0
max_faulty_data = 170.0
log_vars = [
    ("trafo", "in_service"),
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured"),
    ("trafo", "actual_temperature")
]
def add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=80.0, fluctuation=20.0):

    for i, row in net.trafo.iterrows():
        hv_bus = row["hv_bus"]

        sgen_idx = pp.create_sgen(net, bus=hv_bus, p_mw=base_p_mw, q_mvar=0.0, name=f"sgen_trafo_{i}")
        print(f" Created sgen {sgen_idx} at hv_bus {hv_bus} for Trafo {i}")

        time = np.arange(time_steps)
        profile = base_p_mw + fluctuation * np.sin(2 * np.pi * time / time_steps)
        profile = np.clip(profile, 0, None)  # 

        profile_df = pd.DataFrame({"p_mw": profile})
        ds = DFData(profile_df)

        ConstControl(net, element="sgen", variable="p_mw",
                     element_index=[sgen_idx],
                     data_source=ds,
                     profile_name="p_mw")
        print(f" Attached time-varying control to sgen {sgen_idx}")

def inject_transformer_overload_safely(net, time_steps, events_per_trafo=3,
                                       base_load=10.0,
                                       min_factor=2.0, max_factor=3.0,
                                       min_duration=10, max_duration=30):
    for trafo_idx, row in net.trafo.iterrows():
        lv_bus = row["lv_bus"]
        matched_loads = net.load[net.load.bus == lv_bus]

        if matched_loads.empty:
            new_idx = pp.create_load(net, bus=lv_bus, p_mw=base_load, q_mvar=0.0, name=f"synthetic_trafo_{trafo_idx}")
            load_indices = [new_idx]
            print(f" Created synthetic load {new_idx} at lv_bus {lv_bus} for Trafo {trafo_idx}")
        else:
            load_indices = matched_loads.index.tolist()

        profile = np.full(time_steps, base_load)

        for _ in range(events_per_trafo):
            dur = random.randint(min_duration, max_duration)
            start = random.randint(0, time_steps - dur)
            factor = random.uniform(min_factor, max_factor)
            profile[start:start+dur] *= factor
            print(f" Trafo {trafo_idx} overload: t={start}-{start+dur}, factor={factor:.2f}")

        profile_df = pd.DataFrame({"p_mw": profile})

        for load_idx in load_indices:
            ds = pp.timeseries.DFData(profile_df)
            pp.control.ConstControl(
                net, element="load", variable="p_mw",
                element_index=[load_idx],
                data_source=ds,
                profile_name="p_mw"
            )
            print(f" Injected profile to Load {load_idx} for Trafo {trafo_idx}")

def build_test(output_path):
    net = create_30_network()
    trafo_indices = list(net.trafo.index)
    for trafo_idx in trafo_indices:
        print(net.trafo.at[trafo_idx, "lv_bus"])
    add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=30.0, fluctuation=10.0)

    inject_transformer_overload_safely(net, time_steps=time_steps)

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

        # Create the environment
    env = multi_agent_substation_env(net,
                                          trafo_indices=trafo_indices,
                                          delta_p=0.05,
                                          initial_p=0.0,
                                          voltage_tolerance=0.05,
                                          voltage_penalty_factor=10.0,
                                          line_loading_limit=1.0,
                                          power_flow_penalty_factor=5.0,
                                          load_reward_factor=100.0,
                                          transformer_reward_factor=50.0,
                                          disconnection_penalty_factor=50.0,
                                          total_steps=time_steps,
                                          max_temperature=max_temperature)

    # Add the Transformer Protection Controllers with specific FDI lists
    for i, index in enumerate(trafo_indices):
        TransformerDisconnect(
            net=env.net,
            trafo_index=index,
            max_temperature=max_temperature,  # Set the maximum allowable temperature
            T_ambient=T_ambient,  # Ambient temperature
            T_rated=T_rated,  # Rated temperature rise
            n=n,  # Exponent for temperature calculation
            fdi_list=fdi_per_trafo[i],  # Apply specific FDI list for each transformer
            total_steps=time_steps,
            order=1
        )
    for i, index in enumerate(trafo_indices):
        model_path = f"/Users/joshua/PandaPower/tests/dqn_models/trafo_{index}_dqn.pth"
        trafo_controller = DQNTransformerController(
            env,
            trafo_index=index,
            max_temperature=max_temperature,
            # Set the maximum allowable temperature
            T_ambient=T_ambient,  # Ambient temperature
            T_rated=T_rated,  # Rated temperature rise
            n=n,  # Exponent for temperature calculation
            fdi_list=fdi_per_trafo[i],
            total_steps=time_steps,
            model_path=model_path)

    ow = OutputWriter(
        env.net, time_steps=time_steps, output_path=output_path,
        output_file_type='.csv', log_variables=log_vars, csv_separator=';'
    )
    return env, trafo_indices

env, trafo_indices = build_test(output_path)

run_timeseries(env.net, time_steps = range(time_steps))

try:
    transformer_status_log = pd.read_csv(f'{output_path}/trafo/in_service.csv', sep=';')
    transformer_loading = pd.read_csv(f'{output_path}/res_trafo/loading_percent.csv', sep=';')
    transformer_status_log.drop(columns=['Unnamed: 0'], inplace=True)

except FileNotFoundError as e:
    print("File not found. Please ensure the OutputWriter has logged the correct data.")

plot_curves(f"{output_path}/res_trafo/loading_percent.csv", f"{output_path}/loading_curves.png")

# TP, FP, TN, FN
dqn_controllers = [ctrl for ctrl in env.net.controller["object"] if isinstance(ctrl, DQNTransformerController)]

total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

print("**Transformer-Level Confusion Matrix Statistics**")
for controller in dqn_controllers:
    print(f"Transformer {controller.trafo_index}: TP={controller.tp}, FP={controller.fp}, FN={controller.fn}, TN={controller.tn}")
    total_tp += controller.tp
    total_fp += controller.fp
    total_fn += controller.fn
    total_tn += controller.tn

print("\n :")
print(f" True Positive (TP)  = {total_tp}")
print(f" False Positive (FP)  = {total_fp}")
print(f" False Negative (FN)  = {total_fn}")
print(f" True Negative (TN)  = {total_tn}")

conf_matrix = np.array([[total_tp, total_fp], [total_fn, total_tn]])
plot_confusion_matrix(conf_matrix)

