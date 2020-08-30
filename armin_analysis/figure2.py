from pathlib import Path
import numpy as np
from fit_integrator_model import get_model_result, get_target_result
import my_figure as myfig
import matplotlib
import pandas as pd

fig = myfig.Figure(title=f"Figure 2")

experiment = "disc1_hetinx"
coherence_colors = ["#000000", "#330000",  "#990000", "#CC3333"]

root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence")
target_path = Path("/Users/arminbahl/Dropbox/pandas_data_for_ariel/paper/figures")

for data_file in ["all_data.h5", "all_data_best_model_consensus.h5"]:
    if data_file == "all_data.h5":
        ypos = 22
    else:
        ypos = 14

    for genotype in ["wt", "hom"]:
        if genotype == "wt":
            xpos = 2
        else:
            xpos = 9


        ####
        # Show the results from the scn fish
        df_correctness_as_function_of_coherence, \
        df_inter_bout_interval_as_function_of_coherence, \
        df_binned_correctness, \
        df_binned_same_direction, \
        df_binned_features_heading_angle_change_histograms, \
        df_binned_features_inter_bout_interval_histograms, \
        df_gmm_fitting_results = get_target_result(root_path / experiment / data_file, genotype)

        #####
        # Correctness as function of coherence
        p0 = myfig.Plot(fig, num='a1', xpos=xpos, ypos=ypos, plot_height=1.25, plot_width=1.25,
                                       lw=1, pc='white', errorbar_area=False,
                                       xmin=-15, xmax=115, hlines=[50], xticks=[0, 25, 50, 100], xticklabels=[""]*4,
                                       yl="Probability\ncorrect (%)", ymin=44, ymax=91, yticks=[50, 70, 90])

        print(df_correctness_as_function_of_coherence)
        myfig.Line(p0, x=[0, 25, 50, 100], y=df_correctness_as_function_of_coherence.values, lc="black", zorder=1)
        myfig.Scatter(p0, x=[0, 25, 50, 100], y=df_correctness_as_function_of_coherence.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        ####
        # Interbout interval as function of coherence
        p0 = myfig.Plot(fig, num='a2', xpos=xpos, ypos=ypos - 1.5, plot_height=1.25, plot_width=1.25,
                                           lw=1, pc='white', errorbar_area=False,
                                           xl="Coherence (%)", xmin=-15, xmax=115, xticks=[0, 25, 50, 100], hlines=[50],
                                           yl="Interbout\ninterval (s)", ymin=0.45, ymax=1.55, yticks=[0.5, 1.0, 1.5])

        myfig.Line(p0, x=[0, 25, 50, 100], y=df_inter_bout_interval_as_function_of_coherence.values, lc="black", zorder=1)
        myfig.Scatter(p0, x=[0, 25, 50, 100], y=df_inter_bout_interval_as_function_of_coherence.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        ####
        # Correctness over time as function of coherence
        p0 = myfig.Plot(fig, num='a3', xpos=xpos + 2, ypos=ypos - 1.5, plot_height=2.5, title=f"{data_file}, {genotype}",
                                           plot_width=2.2, errorbar_area=False, lw=1,
                                           xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                                           yl="Probability correct (%)", ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])

        for i in range(4):
            data_set = df_binned_correctness.loc[i, :].droplevel(0)
            myfig.Line(p0, data_set.index[1:] - 10, data_set.values[1:], lc=coherence_colors[i], zorder=1)
            myfig.Scatter(p0, x=data_set.index[1:] - 10, y=data_set.values[1:], lc=coherence_colors[i], pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        ####
        # Same direction
        p0 = myfig.Plot(fig, num='a4', xpos=xpos, ypos=ypos - 4, plot_height=1.25, plot_width=1.3,
                                   errorbar_area=False, lw=1,
                                   xl="Time since last turn (s)", xmin=-0.15, xmax=1.6, xticks=[0, 0.5, 1, 1.5], hlines=[50],
                                   yl="Probability to turn\nin same direction (%)", ymin=48, ymax=71,
                                   yticks=[50, 60, 70], vlines=[0])

        myfig.Line(p0, df_binned_same_direction.index, df_binned_same_direction.values, lc="black", zorder=1)
        myfig.Scatter(p0, x=df_binned_same_direction.index, y=df_binned_same_direction.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        ####
        # Turning histograms
        p0 = myfig.Plot(fig, num='a5', xpos=xpos + 2, ypos=ypos - 4, plot_height=1.5, plot_width=1.8,
                                   errorbar_area=False, lw=1,
                                   xl="Turn angle (deg)", xmin=-101, xmax=121, xticks=[-100, -50, 0, 50, 100], hlines=[50],
                                   yl="Probability density", ymin=-0.01, ymax=0.3,
                                   yticks=[], vlines=[0])

        for i in range(4):
            #print("dfgdfg")
            #print(type(df_binned_features_heading_angle_change_histograms))
            #dfgdfgfdgfdg

            data_set = df_binned_features_heading_angle_change_histograms.loc[i, :].droplevel(0).to_frame().query("bin > -101 and bin < 121")

            myfig.Line(p0, data_set.index, data_set.values, lc=coherence_colors[i], zorder=1)



df_estimated_parameters_model1 = pd.read_hdf(root_path / experiment / "estimated_model_parameters.h5", key="data").query("genotype == 'wt'").droplevel(["genotype"])
df_estimated_parameters_model2 = pd.read_hdf(root_path / experiment / "estimated_model_parameters.h5", key="data").query("genotype == 'het'").droplevel(["genotype"])

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






# Tau
p0 = myfig.Plot(fig, num='', xpos=2.0, ypos=7, plot_height=0.75, plot_width=1.25, title="Tau",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=4.1, xticks=[0, 2, 4],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=["WT", "Het"])

myfig.Scatter(p0, x=estimated_tau_model1, y=[0 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_tau_model1)], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_tau_model2, y=[1 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_tau_model2)], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)


# sigma
p0 = myfig.Plot(fig, num='', xpos=4.0, ypos=7, plot_height=0.75, plot_width=1.25, title="Sigma",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=30.1, xticks=[0, 15, 30],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=[""]*2)

myfig.Scatter(p0, x=estimated_noise_sigma_model1, y=[0 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_noise_sigma_model1)], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_noise_sigma_model2, y=[1 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_noise_sigma_model2)], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)



# T
p0 = myfig.Plot(fig, num='', xpos=6.0, ypos=7, plot_height=0.75, plot_width=1.25, title="T",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=2.1, xticks=[0, 1, 2],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=[""]*2)

myfig.Scatter(p0, x=estimated_T_model1, y=[0 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_T_model1)], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_T_model2, y=[1 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_T_model2)], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)


# p_below
p0 = myfig.Plot(fig, num='', xpos=8.0, ypos=7, plot_height=0.75, plot_width=1.25, title="p_below",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.004, xmax=0.061, xticks=[0, 0.02, 0.04, 0.06],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=[""]*2)

myfig.Scatter(p0, x=estimated_bout_clock_probability_below_threshold_model1, y=[0 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_bout_clock_probability_below_threshold_model1)], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_bout_clock_probability_below_threshold_model2, y=[1 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_bout_clock_probability_below_threshold_model2)], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)


# p_above
p0 = myfig.Plot(fig, num='', xpos=10.0, ypos=7, plot_height=0.75, plot_width=1.25, title="p_above",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.004, xmax=0.061, xticks=[0, 0.02, 0.04, 0.06],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=[""]*2)

myfig.Scatter(p0, x=estimated_bout_clock_probability_above_threshold_model1, y=[0 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_bout_clock_probability_above_threshold_model1)], y=[0 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)

myfig.Scatter(p0, x=estimated_bout_clock_probability_above_threshold_model2, y=[1 + 0.2]*12, lc='black', pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[np.median(estimated_bout_clock_probability_above_threshold_model2)], y=[1 - 0.2], lc='C3', pt='o',
                      lw=0.5, ps=9.8, pc='C3', zorder=2, alpha=0.5)






fig.savepdf(target_path / f"raw_figure3_{experiment}", open_pdf=True)
