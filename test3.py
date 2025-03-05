# Reset the network and controllers
import random
import pandas as pd
from pandapower.control import ConstControl
from pandapower.timeseries import run_timeseries, OutputWriter

from controllers.rl_controller import RLController
from controllers.transformer_control import TransformerDisconnect
from envs.substation_env import SubstationEnv
from utils.Generate_fdi import generate_fdi_list
from utils.network import create_network, create_ds

total_episodes = 100
time_steps = 200

net = create_network(time_steps)
# Define transformer indices
trafo_indices = list(net.trafo.index)

# initialise the transformer temperature
net.trafo["temperature_measured"] = 20.0

ds = create_ds(time_steps)

print("Before removal:")
print(net.controller)

# Identify and remove controllers related to Transformers or FDI
controller_to_remove = [idx for idx, ctrl in net.controller.iterrows() if 'Transformer' in str(ctrl['object']) or 'FDI' in str(ctrl['object'])]


# Remove the controller by index
if controller_to_remove:
    net.controller.drop(controller_to_remove, inplace=True)

print("After removal:")
print(net.controller)

T_ambient = 25.0
ΔT_rated = 65.0
n = 1.6
max_temperature = 104.0


# Re-attach the generator control
control = ConstControl(net, element='gen', variable='p_mw', element_index=[0], data_source=ds, profile_name='p_mw')

# Generate FDI lists for each transformer
total_steps = 200
num_attacks = 10
min_faulty_data = 105.0
max_faulty_data = 110.0

fdi_list = generate_fdi_list(total_steps, num_attacks, min_faulty_data, max_faulty_data)

fdi_per_trafo = [[] for _ in trafo_indices]
fdi_attack_log = {}  # Log to track FDI attacks for later analysis

# Distribute FDI attacks across transformers
for fdi in fdi_list:
    target_trafo_index = random.choice(range(len(trafo_indices)))
    fdi_per_trafo[target_trafo_index].append(fdi)
    # Log each FDI attack: {time_step: (trafo_index, faulty_temperature)}
    time_step, faulty_temperature = fdi
    fdi_attack_log[(time_step, target_trafo_index)] = faulty_temperature

# Print the FDI attack log to review all attacks before simulation
for (time_step, trafo_index), faulty_temperature in sorted(fdi_attack_log.items()):
    print(f"Time step {time_step}, Transformer {trafo_index}: Faulty temperature = {faulty_temperature}°C")

# Add the Transformer Protection Controllers with specific FDI lists
for i, index in enumerate(trafo_indices):
    trafo_protection = TransformerDisconnect(
        net=net,
        trafo_index=index,
        max_temperature=max_temperature,      # Set the maximum allowable temperature
        T_ambient=T_ambient,                  # Ambient temperature
        δt_rated=ΔT_rated,                    # Rated temperature rise
        n=n,                                  # Exponent for temperature calculation
        fdi_list=fdi_per_trafo[i],            # Apply specific FDI list for each transformer
        total_steps=total_steps
    )

# Instantiate the environment for the RL controller
env = SubstationEnv(
    net,
    trafo_indices=trafo_indices,
    delta_p=0.05,               # Amount to adjust the disconnection probability
    initial_p=0.0,              # Initial disconnection probability
    voltage_tolerance=0.05,
    voltage_penalty_factor=10.0,
    line_loading_limit=100.0,
    power_flow_penalty_factor=5.0,
    load_reward_factor=20.0,
    transformer_reward_factor=2.0,
    disconnection_penalty_factor=50.0,  # Penalty for high disconnection probabilities
    total_steps=total_steps
)

# Reset the environment
env.reset()

# Initialize the RL controller (it will be automatically added to net.controller)
rl_controller = RLController(env, net, trafo_indices=trafo_indices)

for episode in range(total_episodes):
    print(f"Episode: {episode+1}/{total_episodes}")
    env.reset()
    state = env._get_state()
    # Run the time series simulation
    run_timeseries(net, time_steps=range(total_steps))
    if episode % rl_controller.update_target_every == 0:
        rl_controller.target_net.load_state_dict(rl_controller.policy_net.state_dict())
    rl_controller.log_episode_metrics(episode)
    print(f"Finish Episode {episode+1}/{total_episodes}")


# Specify which variables to record in the output (adding switch closed status)
log_vars = [
    ("trafo", "in_service"),  # Record transformer status
    ("switch", "closed")      # Record switch closed status
]

# Save the transformer and switch status log with a separate output path for exp3
ow = OutputWriter(
    net, time_steps=time_steps, output_path="./output_data_exp3",  # Use exp3 folder
    output_file_type='.csv', log_variables=log_vars, csv_separator=';'
)

# The transformer and switch statuses are stored in output_data_exp3/trafo/in_service.csv and output_data_exp3/switch/closed.csv
try:
    transformer_status_log = pd.read_csv('output_data_exp3/trafo/in_service.csv', sep=';')
    switch_status_log = pd.read_csv('output_data_exp3/switch/closed.csv', sep=';')

    # Clean up the data
    transformer_status_log.drop(columns=['Unnamed: 0'], inplace=True)
    switch_status_log.drop(columns=['Unnamed: 0'], inplace=True)

    # Display the first 10 rows of the transformer and switch status
    print("Transformer In-Service Status:")
    print(transformer_status_log.head(10))

    print("\nSwitch Closed Status:")
    print(switch_status_log.head(10))

except FileNotFoundError as e:
    print("File not found. Please ensure the OutputWriter has logged the correct data.")