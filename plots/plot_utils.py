from pandapower.plotting.plotly import vlevel_plotly
from utils.network import create_network
import plotly.io as pio

fig = vlevel_plotly(create_network())
fig.write_image("pics/voltage_level_plot.png")