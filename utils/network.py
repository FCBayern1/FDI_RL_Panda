import pandapower.networks as pn
import numpy as np
from pandapower.timeseries.data_sources.frame_data import DFData
import pandapower as pp
from pandapower.control import ConstControl
import pandas as pd

def create_network():
    # Create the network
    net = pn.case14()

    for trafo_index in net.trafo.index:
        hv_bus = net.trafo.at[trafo_index, 'hv_bus']
        lv_bus = net.trafo.at[trafo_index, 'lv_bus']

        pp.create_switch(net, bus=hv_bus, element=trafo_index, et='t', closed=True)
        pp.create_switch(net, bus=lv_bus, element=trafo_index, et='t', closed=True)

    return net

def create_ds(time_steps=200):
    # Create a sine wave for generator power
    p_mw = 40 + 1000 * np.sin(np.linspace(0, 2 * np.pi, time_steps))
    df_gen = pd.DataFrame({'p_mw': p_mw}, index=range(time_steps))

    # Attach the constant controller to the power generation
    ds = DFData(df_gen)

    return ds


def create_load_profile(time_steps=200, base_load=50, load_amplitude=30):
    hours = np.linspace(0, 24, time_steps)  # Scale time steps to a 24-hour representation

    # Base sinusoidal load variation (mimicking daily demand cycles)
    daily_variation = load_amplitude * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak at noon, low at night

    # Introduce random overload spikes
    overload_spikes = np.random.choice([0, 10, 20, 30], size=time_steps, p=[0.7, 0.2, 0.08, 0.02])
    # 70% normal, 20% small spikes, 8% moderate overloads, 2% extreme overloads

    # Final load profile: Base load + daily variations + occasional overloads
    dynamic_load_profile = base_load + daily_variation + overload_spikes

    # Ensure the load values remain non-negative
    dynamic_load_profile = np.clip(dynamic_load_profile, 0, None)

    # Create a Pandas DataFrame
    load_profile_df = pd.DataFrame({"p_mw": dynamic_load_profile})

    return load_profile_df


