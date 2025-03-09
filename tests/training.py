import random
import pandas as pd
from pandapower.control import ConstControl
import pandapower as pp
from pandapower.timeseries import run_timeseries, OutputWriter
from controllers.multi_agent_controller import multi_agent_controller
from envs.multi_agent_substation_env import multi_agent_substation_env
from controllers.transformer_control import TransformerDisconnect
from plots.plot_utils import plot_curves
from utils.Generate_fdi import generate_fdi_list
from utils.network import create_network, create_load_profile, create_stable_gen_profile
import torch

total_episodes = 10
time_steps = 100

#Create the network
net = create_network()

# Define transformer indices
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

# load_profiles, overload_steps = create_dynamic_load_profiles(net, time_steps=time_steps, overload_prob=0.1, overload_factor=5.0)
#
# # 绑定负载控制器
# for load_idx, ds in load_profiles.items():
#     ConstControl(net, element="load", variable="p_mw", element_index=[load_idx], data_source=ds, profile_name="p_mw")

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

RLController = multi_agent_controller(env, net, trafo_indices=trafo_indices)

log_vars = [
    ("trafo", "in_service"),  # Record transformer status
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured")
]

ow = OutputWriter(
    net, time_steps=time_steps, output_path="./output_data",
    output_file_type='.csv', log_variables=log_vars, csv_separator=';'
)


for episode in range(total_episodes):
    print(f"Episode: {episode+1}/{total_episodes}")
    env.reset()
    print("Controllers in the network:")
    print(net.controller)
    run_timeseries(net, time_steps = range(time_steps))
    if episode % RLController.update_target_every == 0:
        for idx in trafo_indices:
            RLController.agents[idx]["target_net"].load_state_dict(RLController.agents[idx]["policy_net"].state_dict())
    print(f"Finished Episode {episode+1}/{total_episodes}")

for idx in trafo_indices:
    torch.save(RLController.agents[idx]["policy_net"].state_dict(), f"trafo_{idx}_dqn_model.pth")


try:
    transformer_status_log = pd.read_csv('./output_data/trafo/in_service.csv', sep=';')
    transformer_loading = pd.read_csv('./output_data/res_trafo/loading_percent.csv', sep=';')
    # Clean up the data
    transformer_status_log.drop(columns=['Unnamed: 0'], inplace=True)

except FileNotFoundError as e:
    print("File not found. Please ensure the OutputWriter has logged the correct data.")

plot_curves("./output_data/res_trafo/loading_percent.csv")





