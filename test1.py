import os

import pandas as pd
import numpy as np

import pandapower as pp
import pandapower.networks as pn
from pandapower.control import ConstControl
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.run_time_series import run_timeseries

from controllers.FDIAttackController import FDIAttackController
from controllers.rl_controller import RLController
from controllers.transformer_control import TransformerDisconnect
from envs.substation_env import SubstationEnv

net = pn.case14()

# Define transformer indices
trafo_indices = list(net.trafo.index)

# Define total steps
time_steps = 200

# Create a sine wave for generator power
p_mw = 40 + 500 * np.sin(np.linspace(0, 2 * np.pi, time_steps))
df_gen = pd.DataFrame({'p_mw': p_mw}, index=range(time_steps))

# Attach the constant controller to the power generation
ds = DFData(df_gen)

for trafo_index in net.trafo.index:
    hv_bus = net.trafo.at[trafo_index, 'hv_bus']
    lv_bus = net.trafo.at[trafo_index, 'lv_bus']

    pp.create_switch(net, bus=hv_bus, element=trafo_index, et='t', closed=True)
    pp.create_switch(net, bus=lv_bus, element=trafo_index, et='t', closed=True)

# Reset the network and controllers
print("Before removal:")
print(net.controller)

# Identify and remove controllers related to Transformers or FDI
controller_to_remove = [idx for idx, ctrl in net.controller.iterrows()]

# Remove the controller by index
if controller_to_remove:
    net.controller.drop(controller_to_remove, inplace=True)

print("After removal:")
print(net.controller)

# Re-attach the generator control
control = ConstControl(net, element='gen', variable='p_mw', element_index=[0], data_source=ds, profile_name='p_mw')

# Add the Transformer Protection Controllers with FDI injected manually
target_transformer_index = 0
fault_step = 5
faulty_value = 150.0

for index in trafo_indices:
    if index == target_transformer_index:
        # This transformer will experience an FDI at the specified fault_step
        trafo_protection = FDIAttackController(
            net, trafo_index=index, max_loading_percent=80.0, fault_step=fault_step, faulty_data=faulty_value,
            total_steps=time_steps
        )
    else:
        # Regular transformers without FDI
        trafo_protection = FDIAttackController(
            net, trafo_index=index, max_loading_percent=80.0, total_steps=time_steps
        )

# Instantiate the environment for the RL controller
env = SubstationEnv(
    net,
    trafo_indices=trafo_indices,
    delta_p=0.05,  # Amount to adjust the disconnection probability
    initial_p=0.0,  # Initial disconnection probability
    voltage_tolerance=0.05,
    voltage_penalty_factor=10.0,
    line_loading_limit=100.0,
    power_flow_penalty_factor=5.0,
    load_reward_factor=20.0,
    transformer_reward_factor=2.0,
    disconnection_penalty_factor=50.0,  # Penalty for high disconnection probabilities
    total_steps=time_steps
)

# Reset the environment
env.reset()

# Initialise the RL controller (it will be automatically added to net.controller)
rl_controller = RLController(env, net, trafo_indices=trafo_indices)


output_path = "./output_data_exp3"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Specify which variables to record in the output (adding switch closed status)
log_vars = [
    ("trafo", "in_service"),  # Record transformer status
    ("switch", "closed")  # Record switch closed status
]

# Save the transformer and switch status log with a separate output path for exp3
ow = OutputWriter(
    net, time_steps=time_steps, output_path="./output_data_exp3",  # Use exp3 folder
    output_file_type='.csv', log_variables=log_vars, csv_separator=';'
)


# Run the time series simulation
run_timeseries(net, time_steps=range(time_steps))


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