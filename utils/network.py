import pandapower.networks as pn
import numpy as np
import pandapower as pp
from pandapower.control import ConstControl
import pandas as pd
import random

from pandapower.networks import case_ieee30
from pandapower.timeseries import DFData

def create_network():
    net = pn.case14()

    return net


def create_ds(time_steps=100, base_gen=50, gen_amplitude=50):
    p_mw = base_gen + gen_amplitude * np.sin(np.linspace(0, 2 * np.pi, time_steps))
    df_gen = pd.DataFrame({'p_mw': p_mw}, index=range(time_steps))
    return pp.timeseries.DFData(df_gen)

def create_30_network():
    net = case_ieee30()
    for idx, row in net.trafo.iterrows():
        lv_bus = row["lv_bus"]
        hv_bus = row["hv_bus"]

        connected_loads = net.load[net.load.bus == lv_bus]

        if not connected_loads.empty:
            print(f"🔍 Transformer {idx} supplies bus {lv_bus} with loads. Adding redundancy.")

            candidate_buses = list(set(net.bus.index) - {lv_bus})
            if candidate_buses:
                target_bus = candidate_buses[0]
                pp.create_line_from_parameters(
                    net,
                    from_bus=lv_bus,
                    to_bus=target_bus,
                    length_km=0.1,
                    r_ohm_per_km=0.1,
                    x_ohm_per_km=0.2,
                    c_nf_per_km=10,
                    max_i_ka=0.5,
                    name=f"tie_line_{lv_bus}_{target_bus}"
                )
                print(f"✅ Added tie-line from bus {lv_bus} to bus {target_bus}")
            else:
                pp.create_gen(
                    net,
                    bus=lv_bus,
                    p_mw=0.2,
                    vm_pu=1.02,
                    slack=False,
                    name=f"backup_gen_bus{lv_bus}"
                )
                print(f"⚡ Added backup generator at bus {lv_bus}")
    return net



def create_stable_gen_profile(net, time_steps=100, base_gen_factor=2.0):
    max_load = sum(net.load["p_mw"]) * len(net.gen) * base_gen_factor
    gen_profile = np.full(time_steps, max_load, dtype=float)
    df = pd.DataFrame({"p_mw": gen_profile}, index=range(time_steps))
    return DFData(df)


def create_load_profile(time_steps=100, base_load=60, load_amplitude=30, overload_steps=None, overload_factor=2.0):
    hours = np.linspace(0, 24, time_steps)
    daily_variation = load_amplitude * np.sin(2 * np.pi * (hours - 6) / 24)

    dynamic_load_profile = base_load + daily_variation

    if overload_steps is None:
        overload_steps = random.sample(range(time_steps), int(time_steps * 0.1))
    overload_mask = np.zeros(time_steps, dtype=bool)
    overload_mask[overload_steps] = True

    dynamic_load_profile[overload_mask] *= overload_factor

    noise = np.random.uniform(-5, 5, time_steps)
    dynamic_load_profile = np.clip(dynamic_load_profile + noise, 0, None)

    return pd.DataFrame({"p_mw": dynamic_load_profile})


