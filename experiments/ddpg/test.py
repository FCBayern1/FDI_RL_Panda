import random

import pandas as pd
import numpy as np
from pandapower.control import ConstControl
import pandapower as pp
from pandapower.timeseries import run_timeseries, OutputWriter, DFData

from src.controllers.ddpg_rl_transformer_controller import DDPGTransformerController
from src.envs.DDPG_multi_agent_substation_env import ddpg_multi_agent_substation_env
from src.controllers.transformer_control import TransformerDisconnect
from src.plots.plot_utils import plot_curves, plot_confusion_matrix
from src.utils.Generate_fdi import generate_fdi_list
from src.utils.network import create_stable_gen_profile, create_30_network

# ---------------- 参数配置 ----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
time_steps = 200
T_ambient = 25.0
T_rated = 65.0
n = 1.6
max_temperature = 147.44
model_dir = "./models_ddpg"


def add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=80.0, fluctuation=20.0):

    for i, row in net.trafo.iterrows():
        hv_bus = row["hv_bus"]

        # 创建一个 sgen，初始有功功率为 base_p_mw
        sgen_idx = pp.create_sgen(net, bus=hv_bus, p_mw=base_p_mw, q_mvar=0.0, name=f"sgen_trafo_{i}")
        print(f"⚡ Created sgen {sgen_idx} at hv_bus {hv_bus} for Trafo {i}")

        # 创建时间变化的 profile：base ± fluctuation * sin
        time = np.arange(time_steps)
        profile = base_p_mw + fluctuation * np.sin(2 * np.pi * time / time_steps)
        profile = np.clip(profile, 0, None)  # 保证不为负

        profile_df = pd.DataFrame({"p_mw": profile})
        ds = DFData(profile_df)

        ConstControl(net, element="sgen", variable="p_mw",
                     element_index=[sgen_idx],
                     data_source=ds,
                     profile_name="p_mw")
        print(f"✅ Attached time-varying control to sgen {sgen_idx}")

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
            print(f"➕ Created synthetic load {new_idx} at lv_bus {lv_bus} for Trafo {trafo_idx}")
        else:
            load_indices = matched_loads.index.tolist()

        profile = np.full(time_steps, base_load)

        for _ in range(events_per_trafo):
            dur = random.randint(min_duration, max_duration)
            start = random.randint(0, time_steps - dur)
            factor = random.uniform(min_factor, max_factor)
            profile[start:start+dur] *= factor
            print(f"🔥 Trafo {trafo_idx} overload: t={start}-{start+dur}, factor={factor:.2f}")

        profile_df = pd.DataFrame({"p_mw": profile})

        for load_idx in load_indices:
            ds = pp.timeseries.DFData(profile_df)
            pp.control.ConstControl(
                net, element="load", variable="p_mw",
                element_index=[load_idx],
                data_source=ds,
                profile_name="p_mw"
            )
            print(f"✅ Injected profile to Load {load_idx} for Trafo {trafo_idx}")

# ---------------- 网络构建 ----------------
net = create_30_network()
trafo_indices = list(net.trafo.index)

add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=30.0, fluctuation=10.0)

inject_transformer_overload_safely(net, time_steps=time_steps)

gen_profile = create_stable_gen_profile(net, time_steps=time_steps, base_gen_factor=1.5)

ConstControl(net, element='gen', variable='p_mw', element_index=[0], data_source=gen_profile, profile_name='p_mw',
             order=0)

num_attacks = 20
fdi_list = generate_fdi_list(time_steps, num_attacks, 160, 170.0)
fdi_per_trafo = [[] for _ in trafo_indices]
fdi_attack_log = {}

for fdi in fdi_list:
    target = random.choice(range(len(trafo_indices)))
    fdi_per_trafo[target].append(fdi)
    time_step, faulty_temp = fdi
    fdi_attack_log[(time_step, target)] = faulty_temp

env = ddpg_multi_agent_substation_env(net,
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

for i, idx in enumerate(trafo_indices):
    TransformerDisconnect(
        net=net,
        trafo_index=idx,
        max_temperature=max_temperature,
        T_ambient=T_ambient,
        T_rated=T_rated,
        n=n,
        fdi_list=fdi_per_trafo[i],
        total_steps=time_steps
    )


for i, idx in enumerate(trafo_indices):
    model_path = f"{model_dir}/actor_trafo_{idx}.pth"
    controller = DDPGTransformerController(
        env=env,
        trafo_index=idx,
        max_temperature=max_temperature,
        T_ambient=T_ambient,
        T_rated=T_rated,
        n=n,
        fdi_list=fdi_per_trafo[i],
        total_steps=time_steps,
        model_path=model_path
    )
    print(f"[DEBUG] Actor weights for Trafo {idx}:", controller.actor.state_dict())

log_vars = [
    ("trafo", "in_service"),
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured"),
    ("trafo", "actual_temperature")
]
output_path = "./results_ddpg"
ow = OutputWriter(net, time_steps, output_path, '.csv', log_variables=log_vars, csv_separator=';')

run_timeseries(net, time_steps=range(time_steps))

plot_curves(f"{output_path}/res_trafo/loading_percent.csv", f"{output_path}/loading_curves.png")

# 混淆矩阵统计
ddpg_controllers = [ctrl for ctrl in net.controller["object"] if isinstance(ctrl, DDPGTransformerController)]

total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

print("\n**Transformer-Level Confusion Matrix Statistics**")
for controller in ddpg_controllers:
    controller.print_confusion_matrix()
    total_tp += controller.tp
    total_fp += controller.fp
    total_fn += controller.fn
    total_tn += controller.tn

print("\n✅ 混淆矩阵统计汇总:")
print(f"✅ True Positive (TP) 该断，断了 = {total_tp}")
print(f"❌ False Positive (FP) 不该断，断了 = {total_fp}")
print(f"❌ False Negative (FN) 该断，没断 = {total_fn}")
print(f"✅ True Negative (TN) 不该断，没断 = {total_tn}")

conf_matrix = np.array([[total_tp, total_fp], [total_fn, total_tn]])
plot_confusion_matrix(conf_matrix)
