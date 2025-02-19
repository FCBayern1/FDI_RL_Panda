from pandapower.control.basic_controller import Controller

class FDIAttackController(Controller):
    def __init__(self, net, trafo_index, T_ambient=25.0, ΔT_rated=65.0, n=1.6, fdi_list=None,
                 total_steps=200, in_service=True, order=0, level=0, **kwargs):
        """
        负责模拟 FDI（False Data Injection）攻击，但不控制变压器的 in_service 状态。
        :param net: pandapower 电网模型
        :param trafo_index: 受攻击的变压器索引
        :param T_ambient: 环境温度 (°C)
        :param ΔT_rated: 额定温升 (°C)
        :param n: 温度模型指数
        :param fdi_list: FDI 攻击的时间步列表，格式为 [(time_step, fake_temperature), ...]
        :param total_steps: 总仿真步数
        """
        super().__init__(net, in_service=in_service, order=order, level=level, **kwargs)
        self.net = net
        self.trafo_index = trafo_index
        self.T_ambient = T_ambient
        self.ΔT_rated = ΔT_rated
        self.n = n
        self.fdi_list = fdi_list if fdi_list is not None else []
        self.total_steps = total_steps
        self.current_time_step = None
        self.controller_converged = False

    def calculate_temperature(self, loading_percent):
        """ 根据变压器的负载计算温度 """
        return self.T_ambient + self.ΔT_rated * (loading_percent / 100) ** self.n

    def control_step(self, net):
        """ 只进行 FDI 攻击，不修改变压器的 in_service 状态 """
        if self.controller_converged:
            return

        time_step = self.current_time_step
        if time_step is None:
            return

        # 获取变压器当前的实际负载
        try:
            actual_loading_percent = net.res_trafo.at[self.trafo_index, 'loading_percent']
        except KeyError:
            print(f"⚠️ Time step {time_step}: Transformer {self.trafo_index} - No loading data available.")
            self.controller_converged = True
            return

        # 计算真实温度
        actual_temperature = self.calculate_temperature(actual_loading_percent)
        print(f"🕒 Time step {time_step}: Transformer {self.trafo_index} actual temperature: {actual_temperature:.2f}°C")

        # 检查当前时间步是否有 FDI 攻击
        current_temperature = actual_temperature  # 默认使用真实温度
        for f_step, fake_temperature in self.fdi_list:
            if f_step == time_step:
                current_temperature = fake_temperature  # 伪造温度数据
                print(f"🔴 Time step {time_step}: FDI Injected! Fake temperature for Transformer {self.trafo_index} = {current_temperature:.2f}°C")
                break

        # **这里不再修改 trafo["in_service"]**，仅进行温度欺骗
        print(f"🕒 Time step {time_step}: Transformer {self.trafo_index} reported temperature: {current_temperature:.2f}°C\n")

        self.controller_converged = True  # 本时间步已完成控制

    def time_step(self, net, time):
        """ 进入新的时间步时重置 `controller_converged` 状态 """
        self.current_time_step = time
        self.controller_converged = False

    def is_converged(self, net):
        """ 控制器是否收敛 """
        return self.controller_converged
