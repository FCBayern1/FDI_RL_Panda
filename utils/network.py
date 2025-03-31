import pandapower.networks as pn
import numpy as np
import pandapower as pp
from pandapower.control import ConstControl
import pandas as pd
import random
from pandapower.timeseries import DFData

def create_network():
    # Create the network
    net = pn.case14()

    for trafo_index in net.trafo.index:
        hv_bus = net.trafo.at[trafo_index, 'hv_bus']
        lv_bus = net.trafo.at[trafo_index, 'lv_bus']

        pp.create_switch(net, bus=hv_bus, element=trafo_index, et='t', closed=True)
        pp.create_switch(net, bus=lv_bus, element=trafo_index, et='t', closed=True)

    return net

def create_ds(time_steps=100, base_gen=50, gen_amplitude=50):
    p_mw = base_gen + gen_amplitude * np.sin(np.linspace(0, 2 * np.pi, time_steps))
    df_gen = pd.DataFrame({'p_mw': p_mw}, index=range(time_steps))
    return pp.timeseries.DFData(df_gen)




def create_stable_gen_profile(net, time_steps=100, base_gen_factor=2.0):
    max_load = sum(net.load["p_mw"]) * len(net.gen) * base_gen_factor
    gen_profile = np.full(time_steps, max_load, dtype=float)
    df = pd.DataFrame({"p_mw": gen_profile}, index=range(time_steps))  # 确保列名为 "p_mw"
    return DFData(df)
# def create_load_profile(time_steps=100, base_load=60, load_amplitude=30):
#     hours = np.linspace(0, 24, time_steps)  # Scale time steps to a 24-hour representation
#
#     # Base sinusoidal load variation (mimicking daily demand cycles)
#     daily_variation = load_amplitude * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak at noon, low at night
#
#     # Introduce random overload spikes
#     overload_spikes = np.random.choice([0, 5, 10, 15], size=time_steps, p=[0.7, 0.2, 0.08, 0.02])
#     # 70% normal, 20% small spikes, 8% moderate overloads, 2% extreme overloads
#
#     # Final load profile: Base load + daily variations + occasional overloads
#     dynamic_load_profile = base_load + daily_variation + overload_spikes
#
#     # Ensure the load values remain non-negative
#     dynamic_load_profile = np.clip(dynamic_load_profile, 0, None)
#
#     # Create a Pandas DataFrame
#     load_profile_df = pd.DataFrame({"p_mw": dynamic_load_profile})
#
#     return load_profile_df
#

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

