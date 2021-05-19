from pathlib import Path
import numpy as np
from fit_integrator_model import get_model_result, get_target_result
import my_figure as myfig
import matplotlib
import pandas as pd

fig = myfig.Figure(title=f"Figure 2")

coherence_colors = ["#000000", "#330000",  "#990000", "#CC3333"]

# Show model results, from original surrogte model 2
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence")
#target_path = Path("/Users/arminbahl/Dropbox/pandas_data_for_ariel/paper/figures")

#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence")
#target_path = Path("/Users/arminbahl/Dropbox/pandas_data_for_ariel/paper/figures")
root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/mutant_behavior_data/dot_motion_coherence")
#target_path = Path("/Users/arminbahl/Dropbox/pandas_data_for_ariel/paper/figures")
target_path = Path("/Users/arminbahl/Desktop")

####
# Show the results from the suggorate model
df_correctness_as_function_of_coherence, \
df_inter_bout_interval_as_function_of_coherence, \
df_binned_correctness, \
df_binned_same_direction, \
df_binned_features_heading_angle_change_histograms, \
df_binned_features_inter_bout_interval_histograms, \
df_gmm_fitting_results = get_target_result(root_path / "surrogate_fish1" / "all_data.h5", "wt") # IN REVIEW USE FISH1
#df_gmm_fitting_results = get_target_result(root_path / "surrogate_fish2" / "all_data.h5", "wt")

# Take parameters from model simulation script (this was fish surrogate fish 2)
# tau = 0.8
# noise_sigma = 10
# T = 1
# bout_clock_probability_below_threshold = 0.008
# bout_clock_probability_above_threshold = 0.04
# parameter_title = f"Tau: {tau:.1f}; Sigma: {noise_sigma:.1f}; T: {T:.1f}; p_below: {bout_clock_probability_below_threshold:.4f}; p_above: {bout_clock_probability_above_threshold:.4f}"


# Take parameters from model simulation script (this was fish surrogate fish 1)
tau = 0.8
noise_sigma = 10
T = 1
bout_clock_probability_below_threshold = 0.008
bout_clock_probability_above_threshold = 0.04
parameter_title = f"Tau: {tau:.1f}; Sigma: {noise_sigma:.1f}; T: {T:.1f}; p_below: {bout_clock_probability_below_threshold:.4f}; p_above: {bout_clock_probability_above_threshold:.4f}"


#####
# Correctness as function of coherence
p0 = myfig.Plot(fig, num='b', xpos=5, ypos=22, plot_height=1.25, plot_width=1.25,
                               lw=1, pc='white', errorbar_area=False,
                               xmin=-15, xmax=115, hlines=[50], xticks=[0, 25, 50, 100], xticklabels=[""]*4,
                               yl="Probability\ncorrect (%)", ymin=44, ymax=91, yticks=[50, 70, 90])

print(df_correctness_as_function_of_coherence)
myfig.Line(p0, x=[0, 25, 50, 100], y=df_correctness_as_function_of_coherence.values, lc="black", zorder=1)
myfig.Scatter(p0, x=[0, 25, 50, 100], y=df_correctness_as_function_of_coherence.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

####
# Interbout interval as function of coherence
p0 = myfig.Plot(fig, num='', xpos=5, ypos=20.5, plot_height=1.25, plot_width=1.25,
                                   lw=1, pc='white', errorbar_area=False,
                                   xl="Coherence (%)", xmin=-15, xmax=115, xticks=[0, 25, 50, 100], hlines=[50],
                                   yl="Interbout\ninterval (s)", ymin=0.45, ymax=1.05, yticks=[0.5, 0.75, 1.0])

myfig.Line(p0, x=[0, 25, 50, 100], y=df_inter_bout_interval_as_function_of_coherence.values, lc="black", zorder=1)
myfig.Scatter(p0, x=[0, 25, 50, 100], y=df_inter_bout_interval_as_function_of_coherence.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

####
# Correctness over time as function of coherence
p0 = myfig.Plot(fig, num='c', xpos=8, ypos=20.5, plot_height=2.75, title=parameter_title,
                                   plot_width=2.2, errorbar_area=False, lw=1,
                                   xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                                   yl="Probability correct (%)", ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])

for i in range(4):
    data_set = df_binned_correctness.loc[i, :].droplevel(0)
    myfig.Line(p0, data_set.index - 10, data_set.values, lc=coherence_colors[i], zorder=1)
    myfig.Scatter(p0, x=data_set.index - 10, y=data_set.values, lc=coherence_colors[i], pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

####
# Turning histograms
p0 = myfig.Plot(fig, num='d', xpos=12, ypos=20.5, plot_height=1.5, plot_width=1.8,
                           errorbar_area=False, lw=1,
                           xl="Turn angle (deg)", xmin=-101, xmax=151, xticks=[-100, -50, 0, 50, 100, 150], hlines=[50],
                           yl="Probability density", ymin=-0.01, ymax=0.4,
                           yticks=[], vlines=[0])

for i in range(4):
    data_set = df_binned_features_heading_angle_change_histograms.loc[i,:].droplevel(0)
    print(data_set)
    myfig.Line(p0, data_set.index, data_set.values, lc=coherence_colors[i], zorder=1)

####
# Same direction
p0 = myfig.Plot(fig, num='d', xpos=16, ypos=20.5, plot_height=1.5, plot_width=1.3,
                           errorbar_area=False, lw=1,
                           xl="Time since last turn (s)", xmin=-0.15, xmax=1.6, xticks=[0, 0.5, 1, 1.5], hlines=[50],
                           yl="Probability to turn\nin same direction (%)", ymin=48, ymax=71,
                           yticks=[50, 60, 70], vlines=[0])

myfig.Line(p0, df_binned_same_direction.index, df_binned_same_direction.values, lc="black", zorder=1)
myfig.Scatter(p0, x=df_binned_same_direction.index, y=df_binned_same_direction.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)


####
# Show the results from the fitted model
df_correctness_as_function_of_coherence, \
df_inter_bout_interval_as_function_of_coherence, \
df_binned_correctness, \
df_binned_same_direction, \
df_binned_features_heading_angle_change_histograms, \
df_binned_features_inter_bout_interval_histograms, \
df_gmm_fitting_results = get_target_result(root_path / "surrogate_fish1" / "review1_all_data_best_model_repeat4.h5", "wt")
#get_target_result(root_path / "surrogate_fish2" / "all_data_best_model_repeat1.h5", "wt")

# Get the parameters of the optimized model
#df_estimated_model_parameters = pd.read_hdf(root_path / "surrogate_fish2" / "estimated_model_parameters.h5", key="data")
df_estimated_model_parameters = pd.read_hdf(root_path / "surrogate_fish1" / "review1_estimated_model_parameters.h5", key="data")

#df_estimated_model_parameters = df_estimated_model_parameters.query("repeat == 1 and genotype == 'wt'")
df_estimated_model_parameters = df_estimated_model_parameters.query("repeat == 4 and genotype == 'wt'")
tau = df_estimated_model_parameters['tau'].values[0]
noise_sigma = df_estimated_model_parameters['noise_sigma'].values[0]
T = df_estimated_model_parameters['T'].values[0]
bout_clock_probability_below_threshold = df_estimated_model_parameters['bout_clock_probability_below_threshold'].values[0]
bout_clock_probability_above_threshold = df_estimated_model_parameters['bout_clock_probability_above_threshold'].values[0]
parameter_title = f"Tau: {tau:.1f}; Sigma: {noise_sigma:.1f}; T: {T:.1f}; p_below: {bout_clock_probability_below_threshold:.4f}; p_above: {bout_clock_probability_above_threshold:.4f}"


#####
# Correctness as function of coherence
p0 = myfig.Plot(fig, num='b', xpos=5, ypos=17, plot_height=1.25, plot_width=1.25,
                               lw=1, pc='white', errorbar_area=False,
                               xmin=-15, xmax=115, hlines=[50], xticks=[0, 25, 50, 100], xticklabels=[""]*4,
                               yl="Probability\ncorrect (%)", ymin=44, ymax=91, yticks=[50, 70, 90])

myfig.Line(p0, x=[0, 25, 50, 100], y=df_correctness_as_function_of_coherence.values, lc="black", zorder=1)
myfig.Scatter(p0, x=[0, 25, 50, 100], y=df_correctness_as_function_of_coherence.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

####
p0 = myfig.Plot(fig, num='', xpos=5, ypos=15.5, plot_height=1.25, plot_width=1.25,
                                   lw=1, pc='white', errorbar_area=False,
                                   xl="Coherence (%)", xmin=-15, xmax=115, xticks=[0, 25, 50, 100], hlines=[50],
                                   yl="Interbout\ninterval (s)", ymin=0.45, ymax=1.05, yticks=[0.5, 0.75, 1.0])

myfig.Line(p0, x=[0, 25, 50, 100], y=df_inter_bout_interval_as_function_of_coherence.values, lc="black", zorder=1)
myfig.Scatter(p0, x=[0, 25, 50, 100], y=df_inter_bout_interval_as_function_of_coherence.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)


####
# Correctness over time as function of coherence
p0 = myfig.Plot(fig, num='c', xpos=8, ypos=15.5, plot_height=2.75, title = parameter_title,
                                   plot_width=2.2, errorbar_area=False, lw=1,
                                   xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                                   yl="Probability correct (%)", ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])


for i in range(4):
    data_set = df_binned_correctness.loc[i, :].droplevel(0)
    myfig.Line(p0, data_set.index - 10, data_set.values, lc=coherence_colors[i], zorder=1)
    myfig.Scatter(p0, x=data_set.index - 10, y=data_set.values, lc=coherence_colors[i], pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

####
# Turning histograms
p0 = myfig.Plot(fig, num='d', xpos=12, ypos=15.5, plot_height=1.5, plot_width=1.8,
                           errorbar_area=False, lw=1,
                           xl="Turn angle (deg)", xmin=-101, xmax=151, xticks=[-100, -50, 0, 50, 100, 150], hlines=[50],
                           yl="Probability density", ymin=-0.01, ymax=0.4,
                           yticks=[], vlines=[0])

for i in range(4):
    data_set = df_binned_features_heading_angle_change_histograms.loc[i,:].droplevel(0)
    print(data_set)
    myfig.Line(p0, data_set.index, data_set.values, lc=coherence_colors[i], zorder=1)

####
# Same direction
p0 = myfig.Plot(fig, num='d', xpos=16, ypos=15.5, plot_height=1.5, plot_width=1.3,
                           errorbar_area=False, lw=1,
                           xl="Time since last turn (s)", xmin=-0.15, xmax=1.6, xticks=[0, 0.5, 1, 1.5], hlines=[50],
                           yl="Probability to turn\nin same direction (%)", ymin=48, ymax=71,
                           yticks=[50, 60, 70], vlines=[0])

myfig.Line(p0, df_binned_same_direction.index, df_binned_same_direction.values, lc="black", zorder=1)
myfig.Scatter(p0, x=df_binned_same_direction.index, y=df_binned_same_direction.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

review_string = "review1_"

####
# Show the different parameter estimation for different repeats and models
df_estimated_parameters_model1 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}estimated_model_parameters.h5", key="data").query("genotype == 'wt'").droplevel(["genotype"])
df_estimated_parameters_model2 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}estimated_model_parameters.h5", key="data").query("genotype == 'wt'").droplevel(["genotype"])
df_estimated_parameters_model3 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}estimated_model_parameters.h5", key="data").query("genotype == 'wt'").droplevel(["genotype"])
df_estimated_parameters_model4 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}estimated_model_parameters.h5", key="data").query("genotype == 'wt'").droplevel(["genotype"])

df_minimal_error_gen0_model1 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 0").droplevel(["genotype", "generation"])
df_minimal_error_gen0_model2 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 0").droplevel(["genotype", "generation"])
df_minimal_error_gen0_model3 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 0").droplevel(["genotype", "generation"])
df_minimal_error_gen0_model4 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 0").droplevel(["genotype", "generation"])

df_minimal_error_gen79_model1 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 79").droplevel(["genotype", "generation"])
df_minimal_error_gen79_model2 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 79").droplevel(["genotype", "generation"])
df_minimal_error_gen79_model3 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 79").droplevel(["genotype", "generation"])
df_minimal_error_gen79_model4 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 79").droplevel(["genotype", "generation"])

#
# df_estimated_parameters_model1 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}estimated_model_parameters.h5", key="data").query("genotype == 'wt'").droplevel(["genotype"])
# df_estimated_parameters_model2 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}estimated_model_parameters.h5", key="data").query("genotype == 'het'").droplevel(["genotype"])
# df_estimated_parameters_model3 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}estimated_model_parameters.h5", key="data").query("genotype == 'hom'").droplevel(["genotype"])
# df_estimated_parameters_model4 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}estimated_model_parameters.h5", key="data").query("genotype == 'hom'").droplevel(["genotype"])
#
# df_minimal_error_gen0_model1 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 0").droplevel(["genotype", "generation"])
# df_minimal_error_gen0_model2 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'het' and generation == 0").droplevel(["genotype", "generation"])
# df_minimal_error_gen0_model3 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'hom' and generation == 0").droplevel(["genotype", "generation"])
# df_minimal_error_gen0_model4 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'hom' and generation == 0").droplevel(["genotype", "generation"])
#
# df_minimal_error_gen79_model1 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 79").droplevel(["genotype", "generation"])
# df_minimal_error_gen79_model2 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'het' and generation == 79").droplevel(["genotype", "generation"])
# df_minimal_error_gen79_model3 = pd.read_hdf(root_path / "surrogate_fish2" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'hom' and generation == 79").droplevel(["genotype", "generation"])
# df_minimal_error_gen79_model4 = pd.read_hdf(root_path / "surrogate_fish1" / f"{review_string}errors_over_generations.h5", key="data").query("genotype == 'hom' and generation == 79").droplevel(["genotype", "generation"])

#####
# Display the evolution of error functions over generation
#df_errors_over_generations = pd.read_hdf(root_path / "surrogate_fish2" / "errors_over_generations.h5", key="data")
df_errors_over_generations = pd.read_hdf(root_path / "surrogate_fish1" / "errors_over_generations.h5", key="data")
df_errors_over_generations = df_errors_over_generations.query("repeat == 1 and genotype == 'wt'").droplevel(["repeat", "genotype"])

p0 = myfig.Plot(fig, num='b', xpos=2, ypos=10, plot_height=3, plot_width=1.5, title="Model 1, repeat 1",
                               lw=1, pc='white', errorbar_area=False,
                               xl="Generation", xmin=-1, xmax=81, xticks=[0, 20, 40, 60, 80],
                               yl="Minimal error\nin population", ymin=-0.01, ymax=0.31, yticks=[0, 0.15, 0.3], hlines=[0])

for error_i in range(6):
    myfig.Line(p0, x=df_errors_over_generations.loc[error_i, :].index, y=df_errors_over_generations.loc[error_i, :].values, lc=f"C{error_i}", zorder=1, label=f"Error {error_i}")
    myfig.Scatter(p0, x=df_errors_over_generations.loc[error_i, :].index, y=df_errors_over_generations.loc[error_i, :].values, lc=f"C{error_i}", pt='o', lw=0.5, ps=1, pc='white', zorder=2)

####
# Minimal error
p0 = myfig.Plot(fig, num='', xpos=5, ypos=10, plot_height=3, plot_width=1.25, title="First generation:\nMinimal error\nin population",
            lw=1, pc='white', errorbar_area=False,
            xl="Error value", xmin=-0.05, xmax=0.65, xticks=[0, 0.3, 0.6],
            yl="", ymin=-0.6, ymax=3.6, yticks=[0, 1, 2, 3], yticklabels=["Model 1", "Model 2", "Model 3", "Model 4"])

for error_i in range(6):
    myfig.Scatter(p0, x=df_minimal_error_gen0_model1.query("error_i == @error_i").values, y=[0 + (error_i - 2.5)/8] * 5, lc=f'C{error_i}', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
    myfig.Scatter(p0, x=df_minimal_error_gen0_model2.query("error_i == @error_i").values, y=[1 + (error_i - 2.5)/8] * 5, lc=f'C{error_i}', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
    myfig.Scatter(p0, x=df_minimal_error_gen0_model3.query("error_i == @error_i").values, y=[2 + (error_i - 2.5)/8] * 5, lc=f'C{error_i}', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
    myfig.Scatter(p0, x=df_minimal_error_gen0_model4.query("error_i == @error_i").values, y=[3 + (error_i - 2.5)/8] * 5, lc=f'C{error_i}', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)

p0 = myfig.Plot(fig, num='', xpos=7, ypos=10, plot_height=3, plot_width=1.25, title="Last generation:\nMinimal error\nin population",
            lw=1, pc='white', errorbar_area=False,
            xl="Error value", xmin=-0.05, xmax=0.65, xticks=[0, 0.3, 0.6],
            yl="", ymin=-0.6, ymax=3.6, yticks=[0, 1, 2, 3], yticklabels=[""]*4)

for error_i in range(6):
    myfig.Scatter(p0, x=df_minimal_error_gen79_model1.query("error_i == @error_i").values, y=[0 + (error_i - 2.5)/8] * 5, lc=f'C{error_i}', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
    myfig.Scatter(p0, x=df_minimal_error_gen79_model2.query("error_i == @error_i").values, y=[1 + (error_i - 2.5)/8] * 5, lc=f'C{error_i}', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
    myfig.Scatter(p0, x=df_minimal_error_gen79_model3.query("error_i == @error_i").values, y=[2 + (error_i - 2.5)/8] * 5, lc=f'C{error_i}', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
    myfig.Scatter(p0, x=df_minimal_error_gen79_model4.query("error_i == @error_i").values, y=[3 + (error_i - 2.5)/8] * 5, lc=f'C{error_i}', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)


######
# parameters found
estimated_tau_model1 = df_estimated_parameters_model1['tau'].values
estimated_noise_sigma_model1 = df_estimated_parameters_model1['noise_sigma']
estimated_T_model1 = df_estimated_parameters_model1['T']
estimated_bout_clock_probability_below_threshold_model1 = df_estimated_parameters_model1['bout_clock_probability_below_threshold']
estimated_bout_clock_probability_above_threshold_model1 = df_estimated_parameters_model1['bout_clock_probability_above_threshold']

estimated_tau_model2 = df_estimated_parameters_model2['tau'].values
estimated_noise_sigma_model2 = df_estimated_parameters_model2['noise_sigma']
estimated_T_model2 = df_estimated_parameters_model2['T']
estimated_bout_clock_probability_below_threshold_model2 = df_estimated_parameters_model2['bout_clock_probability_below_threshold']
estimated_bout_clock_probability_above_threshold_model2 = df_estimated_parameters_model2['bout_clock_probability_above_threshold']

estimated_tau_model3 = df_estimated_parameters_model3['tau'].values
estimated_noise_sigma_model3 = df_estimated_parameters_model3['noise_sigma']
estimated_T_model3 = df_estimated_parameters_model3['T']
estimated_bout_clock_probability_below_threshold_model3 = df_estimated_parameters_model3['bout_clock_probability_below_threshold']
estimated_bout_clock_probability_above_threshold_model3 = df_estimated_parameters_model3['bout_clock_probability_above_threshold']

estimated_tau_model4 = df_estimated_parameters_model4['tau'].values
estimated_noise_sigma_model4 = df_estimated_parameters_model4['noise_sigma']
estimated_T_model4 = df_estimated_parameters_model4['T']
estimated_bout_clock_probability_below_threshold_model4 = df_estimated_parameters_model4['bout_clock_probability_below_threshold']
estimated_bout_clock_probability_above_threshold_model4 = df_estimated_parameters_model4['bout_clock_probability_above_threshold']

# Tau
p0 = myfig.Plot(fig, num='', xpos=2.0, ypos=7, plot_height=1.25, plot_width=1.25, title="Tau",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=4.1, xticks=[0, 1, 2, 3, 4],
            yl="", ymin=-0.6, ymax=3.6, yticks=[0, 1, 2, 3], yticklabels=["Model 1", "Model 2", "Model 3", "Model 4"])

myfig.Scatter(p0, x=estimated_tau_model1, y=[0 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.8], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_tau_model2, y=[1 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.8], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_tau_model3, y=[2 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.8], y=[2 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_tau_model4, y=[3 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[2.5], y=[3 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

# sigma
p0 = myfig.Plot(fig, num='', xpos=4.0, ypos=7, plot_height=1.25, plot_width=1.25, title="Sigma",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=30.1, xticks=[0, 15, 30],
            yl="", ymin=-0.6, ymax=3.6, yticks=[0, 1, 2, 3], yticklabels=[""]*4)

myfig.Scatter(p0, x=estimated_noise_sigma_model1, y=[0 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[10], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_noise_sigma_model2, y=[1 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[15], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_noise_sigma_model3, y=[2 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[20], y=[2 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_noise_sigma_model4, y=[3 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[10], y=[3 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)


# T
p0 = myfig.Plot(fig, num='', xpos=6.0, ypos=7, plot_height=1.25, plot_width=1.25, title="T",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=4.1, xticks=[0, 1, 2, 3, 4],
            yl="", ymin=-0.6, ymax=3.6, yticks=[0, 1, 2, 3], yticklabels=[""]*4)

myfig.Scatter(p0, x=estimated_T_model1, y=[0 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[1], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_T_model2, y=[1 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[1], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_T_model3, y=[2 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[1], y=[2 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_T_model4, y=[3 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[1], y=[3 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)


# p_below
p0 = myfig.Plot(fig, num='', xpos=8.0, ypos=7, plot_height=1.25, plot_width=1.25, title="p_below",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=0.004, xmax=0.021, xticks=[0.005, 0.01, 0.015, 0.02],
            yl="", ymin=-0.6, ymax=3.6, yticks=[0, 1, 2, 3], yticklabels=[""]*4)

myfig.Scatter(p0, x=estimated_bout_clock_probability_below_threshold_model1, y=[0 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.008], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_bout_clock_probability_below_threshold_model2, y=[1 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.008], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_bout_clock_probability_below_threshold_model3, y=[2 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.008], y=[2 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_bout_clock_probability_below_threshold_model4, y=[3 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.008], y=[3 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

# p_above
p0 = myfig.Plot(fig, num='', xpos=10.0, ypos=7, plot_height=1.25, plot_width=1.25, title="p_above",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=0.019, xmax=0.081, xticks=[0.02, 0.04, 0.06, 0.08],
            yl="", ymin=-0.6, ymax=3.6, yticks=[0, 1, 2, 3], yticklabels=[""]*4)

myfig.Scatter(p0, x=estimated_bout_clock_probability_above_threshold_model1, y=[0 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.04], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_bout_clock_probability_above_threshold_model2, y=[1 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.04], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_bout_clock_probability_above_threshold_model3, y=[2 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.04], y=[2 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_bout_clock_probability_above_threshold_model4, y=[3 + 0.2]*5, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[0.04], y=[3 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)


fig.savepdf(target_path / f"raw_figure2", open_pdf=True)
