from plots.plot_utils import plot_temperature, plot_service, plot_curves

file_path_0 = "/Users/joshua/PandaPower/tests/output_data/trafo/temperature_measured.csv"
file_path_1 = "/Users/joshua/PandaPower/tests/output_data/trafo/in_service.csv"
file_path_2 = "/Users/joshua/PandaPower/tests/output_data/res_trafo/loading_percent.csv"
file_path_3 = "/Users/joshua/PandaPower/tests/output_data/trafo/actual_temperature.csv"

output_path_0 = "/Users/joshua/PandaPower/plots/pics/temperature_curve.png"
output_path_1 = "/Users/joshua/PandaPower/plots/pics/in_service.png"
output_path_2 = "/Users/joshua/PandaPower/plots/pics/loading.png"
output_path_3 = "/Users/joshua/PandaPower/plots/pics/actual_temperature.png"

plot_temperature(file_path_0, output_path_0)
plot_service(file_path_1, output_path_1)
plot_curves(file_path_2,output_path_2)
plot_temperature(file_path_3, output_path_3)