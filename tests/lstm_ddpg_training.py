import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandapower.control import ConstControl
import pandapower as pp
from pandapower.timeseries import run_timeseries, OutputWriter, DFData

from controllers.transformer_control import TransformerDisconnect
from models.DDPG import MultiAgentDDPGTrainer

from utils.Generate_fdi import generate_fdi_list
from utils.network import create_network, create_load_profile, create_stable_gen_profile

from controllers.ddpg_multi_agent_controller import DDPGMultiAgentController
from envs.DDPG_multi_agent_substation_env import ddpg_multi_agent_substation_env

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

total_episodes = 15
time_steps = 200
T_ambient = 25.0
T_rated = 65.0
n = 1.6
max_temperature = 147.44
# Generate FDI lists for each transformer
num_attacks = 20
min_faulty_data = 95.0
max_faulty_data = 100.0
log_vars = [
    ("trafo", "in_service"),
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured"),
    ("trafo", "actual_temperature")
]

metrics_log = {
    "episode": [],
    "TP": [],
    "FP": [],
    "FN": [],
    "TN": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "accuracy": []
}