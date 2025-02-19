import pandapower.networks as pn
import numpy as np
from pandapower.timeseries.data_sources.frame_data import DFData
import pandapower as pp
from pandapower.control import ConstControl
import pandas as pd

def create_network(time_steps = 200):
    # Create the network
    net = pn.case14()

    # Attach the constant controller to the power generation
    ds = create_ds(time_steps)

    for trafo_index in net.trafo.index:
        hv_bus = net.trafo.at[trafo_index, 'hv_bus']
        lv_bus = net.trafo.at[trafo_index, 'lv_bus']

        pp.create_switch(net, bus=hv_bus, element=trafo_index, et='t', closed=True)
        pp.create_switch(net, bus=lv_bus, element=trafo_index, et='t', closed=True)

    control = ConstControl(net, element='gen', variable='p_mw', element_index=[0], data_source=ds, profile_name='p_mw')
    return net

def create_ds(time_steps=200):
    # Create a sine wave for generator power
    p_mw = 40 + 1000 * np.sin(np.linspace(0, 2 * np.pi, time_steps))
    df_gen = pd.DataFrame({'p_mw': p_mw}, index=range(time_steps))

    # Attach the constant controller to the power generation
    ds = DFData(df_gen)

    return ds