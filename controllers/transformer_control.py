import numpy as np
from pandapower.control.basic_controller import Controller


class TransformerDisconnect(Controller):
    def __init__(self, net, trafo_index, max_temperature, T_ambient=25.0, T_rated=65.0, n=1.6, fdi_list=None,
                 total_steps=200, in_service=True, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level, **kwargs)
        self.net = net
        self.trafo_index = trafo_index
        self.max_temperature = max_temperature
        self.T_ambient = T_ambient
        self.T_rated = T_rated
        self.n = n
        self.fdi_list = fdi_list if fdi_list is not None else []
        self.total_steps = total_steps
        self.current_time_step = None
        self.trafo_disconnected = False
        self.controller_converged = False

    def calculate_temperature(self, loading_percent):
        # Calculate the transformer temperature based on loading percent
        return self.T_ambient + self.T_rated * (loading_percent) ** self.n

    def control_step(self, net):
        if self.controller_converged:
            return

        time_step = self.current_time_step
        if time_step is None:
            return

        # Get actual loading percent of the transformer before any FDI injection
        try:
            actual_loading_percent = np.nan_to_num(net.res_trafo.at[self.trafo_index, 'loading_percent'],0.0)
        except KeyError:
            print(f"Time step {time_step}: KeyError - No data available for transformer at index {self.trafo_index}")
            self.controller_converged = True
            return

        actual_temperature = self.calculate_temperature(actual_loading_percent)
        self.net.trafo.at[self.trafo_index, 'temperature_measured'] = actual_temperature
        print(f"\n Time step {time_step}: The actual temperature of transformer {self.trafo_index} is {actual_temperature:.2f}°C, actual loading percent is {actual_loading_percent:.2f}")

        # Check if an FDI attack should be applied at this specific time step for this transformer
        current_temperature = actual_temperature
        for f_step, faulty_temperature in self.fdi_list:
            if f_step == time_step:
                self.net.trafo.at[self.trafo_index, 'temperature_measured'] = faulty_temperature
                self.net.trafo.at[self.trafo_index, 'fdi'] = True
                current_temperature = faulty_temperature
                print(
                    f"🌹🌹Time step {time_step}: FDI Injected, setting trafo {self.trafo_index} temperature to {faulty_temperature}°C")
                break

        # If no FDI attack is specified for this time step, use the actual temperature data
        if self.net.trafo.at[self.trafo_index, 'temperature_measured'] is None:
            self.net.trafo.at[self.trafo_index, 'temperature_measured'] = actual_temperature

        print(f"Time step {time_step}: Transformer {self.trafo_index} current reading: {current_temperature:.2f}°C")

        # Decide whether to disconnect the transformer based on the temperature
        if self.net.trafo.at[self.trafo_index, 'temperature_measured'] > self.max_temperature and not self.trafo_disconnected:
            net.trafo.at[self.trafo_index, "in_service"] = False
        else:
            net.trafo.at[self.trafo_index, "in_service"] = True
        self.controller_converged = True

    def time_step(self, net, time):
        self.current_time_step = time
        self.controller_converged = False  # Reset convergence at the beginning of each time step

    def is_converged(self, net):
        return self.controller_converged
