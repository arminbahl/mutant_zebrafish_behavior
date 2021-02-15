from pathlib import Path
import numpy as np
from fit_integrator_model import get_model_result, get_target_result
import my_figure as myfig
import matplotlib
from matplotlib import colors
import pandas as pd

fig = myfig.Figure(title=f"Figure 2")

# experiment = "scn1lab_NIBR_20200708"
# mutant_name= 'het'
# basecolor = "C3"

experiment = "scn1lab_zirc_20200710"
mutant_name= 'het'
basecolor = "red"

# experiment = "disc1_hetinx"
# mutant_name= 'hom'
basecolor = "C4"
#/mutant_behavior_data
root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/mutant_behavior_data/dot_motion_coherence")
target_path = Path("/Users/arminbahl/Dropbox/pandas_data_for_ariel/paper/figures")

df_estimated_parameters_wt = pd.read_hdf(root_path / experiment / "estimated_model_parameters.h5", key="data").query("genotype == 'wt'").droplevel(["genotype"])
df_estimated_parameters_mutant = pd.read_hdf(root_path / experiment / "estimated_model_parameters.h5", key="data").query("genotype == @mutant_name").droplevel(["genotype"])

df_minimal_error_gen0_wt = pd.read_hdf(root_path / experiment / "errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 0").droplevel(["genotype", "generation"])
df_minimal_error_gen0_mutant = pd.read_hdf(root_path / experiment / "errors_over_generations.h5", key="data").query("genotype == @mutant_name and generation == 0").droplevel(["genotype", "generation"])

df_minimal_error_gen79_wt = pd.read_hdf(root_path / experiment / "errors_over_generations.h5", key="data").query("genotype == 'wt' and generation == 79").droplevel(["genotype", "generation"])
df_minimal_error_gen79_mutant = pd.read_hdf(root_path / experiment / "errors_over_generations.h5", key="data").query("genotype == @mutant_name and generation == 79").droplevel(["genotype", "generation"])

best_repeat_model_wt = df_minimal_error_gen79_wt.query("error_i == 5")["error"].argmin()
best_repeat_model_mutant = df_minimal_error_gen79_mutant.query("error_i == 5")["error"].argmin()

for data_model in [0, 1]:

    if data_model == 0:
        ypos = 22
        df_extracted_features_wt = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_features")
        df_extracted_binned_features_wt = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features")
        df_extracted_binned_features_same_direction_wt = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_same_direction")
        df_extracted_binned_features_heading_angle_change_histograms_wt = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_heading_angle_change_histograms")
        df_extracted_binned_features_inter_bout_interval_histograms_wt = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_inter_bout_interval_histograms")

        df_extracted_features_mutant = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_features")
        df_extracted_binned_features_mutant = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features")
        df_extracted_binned_features_same_direction_mutant = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_same_direction")
        df_extracted_binned_features_heading_angle_change_histograms_mutant = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_heading_angle_change_histograms")
        df_extracted_binned_features_inter_bout_interval_histograms_mutant = pd.read_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_inter_bout_interval_histograms")

        print("wt N = ", len(df_extracted_binned_features_wt.query("genotype == 'wt' and stim == 0 and bin == 7")))
        print("mutant N = ", len(df_extracted_binned_features_mutant.query("genotype == @mutant_name and stim == 0 and bin == 7")))

    sdf
    if data_model == 1:
        ypos = 14
        df_extracted_features_wt = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_wt}.h5", key="extracted_features")
        df_extracted_binned_features_wt = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_wt}.h5", key="extracted_binned_features")
        df_extracted_binned_features_same_direction_wt = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_wt}.h5", key="extracted_binned_features_same_direction")
        df_extracted_binned_features_heading_angle_change_histograms_wt = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_wt}.h5", key="extracted_binned_features_heading_angle_change_histograms")
        df_extracted_binned_features_inter_bout_interval_histograms_wt = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_wt}.h5", key="extracted_binned_features_inter_bout_interval_histograms")

        df_extracted_features_mutant = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_mutant}.h5", key="extracted_features")
        df_extracted_binned_features_mutant = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_mutant}.h5", key="extracted_binned_features")
        df_extracted_binned_features_same_direction_mutant = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_mutant}.h5", key="extracted_binned_features_same_direction")
        df_extracted_binned_features_heading_angle_change_histograms_mutant = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_mutant}.h5", key="extracted_binned_features_heading_angle_change_histograms")
        df_extracted_binned_features_inter_bout_interval_histograms_mutant = pd.read_hdf(root_path / experiment / f"all_data_best_model_repeat{best_repeat_model_mutant}.h5", key="extracted_binned_features_inter_bout_interval_histograms")


    correctness_as_function_of_coherence_wt_mean = df_extracted_features_wt.query("genotype == 'wt'").groupby("stim").mean()["correctness"]
    correctness_as_function_of_coherence_wt_sem = df_extracted_features_wt.query("genotype == 'wt'").groupby("stim").sem()["correctness"]

    interbout_interval_as_function_of_coherence_wt_mean = (df_extracted_features_wt.query("genotype == 'wt'")["inter_bout_interval"]).groupby("stim").mean()
    interbout_interval_as_function_of_coherence_wt_sem = (df_extracted_features_wt.query("genotype == 'wt'")["inter_bout_interval"]).groupby("stim").sem()

    binned_correctness_wt_mean = df_extracted_binned_features_wt.query("genotype == 'wt'").groupby(["stim", "bin"]).mean()
    binned_correctness_wt_sem = df_extracted_binned_features_wt.query("genotype == 'wt'").groupby(["stim", "bin"]).sem()

    binned_same_direction_wt_mean = df_extracted_binned_features_same_direction_wt.query("genotype == 'wt'").groupby(["bin"]).mean()
    binned_same_direction_wt_sem = df_extracted_binned_features_same_direction_wt.query("genotype == 'wt'").groupby(["bin"]).sem()

    histogram_heading_angle_change_wt_mean = df_extracted_binned_features_heading_angle_change_histograms_wt.query("genotype == 'wt'").groupby(["stim", "bin"]).mean()
    histogram_heading_angle_change_wt_sem = df_extracted_binned_features_heading_angle_change_histograms_wt.query("genotype == 'wt'").groupby(["stim", "bin"]).sem()

    correctness_as_function_of_coherence_mutant_mean = df_extracted_features_mutant.query("genotype == @mutant_name").groupby("stim").mean()["correctness"]
    correctness_as_function_of_coherence_mutant_sem = df_extracted_features_mutant.query("genotype == @mutant_name").groupby("stim").sem()["correctness"]

    interbout_interval_as_function_of_coherence_mutant_mean = (df_extracted_features_mutant.query("genotype == @mutant_name")["inter_bout_interval"]).groupby("stim").mean()
    interbout_interval_as_function_of_coherence_mutant_sem = (df_extracted_features_mutant.query("genotype == @mutant_name")["inter_bout_interval"]).groupby("stim").sem()

    binned_correctness_mutant_mean = df_extracted_binned_features_mutant.query("genotype == @mutant_name").groupby(["stim", "bin"]).mean()
    binned_correctness_mutant_sem = df_extracted_binned_features_mutant.query("genotype == @mutant_name").groupby(["stim", "bin"]).sem()

    binned_same_direction_mutant_mean = df_extracted_binned_features_same_direction_mutant.query("genotype == @mutant_name").groupby(["bin"]).mean()
    binned_same_direction_mutant_sem = df_extracted_binned_features_same_direction_mutant.query("genotype == @mutant_name").groupby(["bin"]).sem()

    histogram_heading_angle_change_mutant_mean = df_extracted_binned_features_heading_angle_change_histograms_mutant.query("genotype == @mutant_name").groupby(["stim", "bin"]).mean()
    histogram_heading_angle_change_mutant_sem = df_extracted_binned_features_heading_angle_change_histograms_mutant.query("genotype == @mutant_name").groupby(["stim", "bin"]).sem()

    #####
    # Correctness as function of coherence
    p0 = myfig.Plot(fig, num='b1', xpos=10, ypos=ypos, plot_height=1.25, plot_width=1.25, title="Wildtype / mutant",
                                   lw=1, pc='white', errorbar_area=False,
                                   xmin=-15, xmax=115, hlines=[50], xticks=[0, 25, 50, 100], xticklabels=[""]*4,
                                   yl="Probability\ncorrect (%)", ymin=44, ymax=91, yticks=[50, 70, 90])

    myfig.Line(p0, x=[0, 25, 50, 100], y=correctness_as_function_of_coherence_wt_mean, yerr=correctness_as_function_of_coherence_wt_sem, lc="black", zorder=1)
    myfig.Scatter(p0, x=[0, 25, 50, 100], y=correctness_as_function_of_coherence_wt_mean, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

    myfig.Line(p0, x=[0, 25, 50, 100], y=correctness_as_function_of_coherence_mutant_mean, yerr=correctness_as_function_of_coherence_mutant_sem, lc=basecolor, zorder=1)
    myfig.Scatter(p0, x=[0, 25, 50, 100], y=correctness_as_function_of_coherence_mutant_mean, lc=basecolor, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

    ####
    # Interbout interval as function of coherence
    p0 = myfig.Plot(fig, num='b2', xpos=10, ypos=ypos - 1.5, plot_height=1.25, plot_width=1.25,
                                       lw=1, pc='white', errorbar_area=False,
                                       xl="Coherence (%)", xmin=-15, xmax=115, xticks=[0, 25, 50, 100], hlines=[50],
                                       yl="Interbout\ninterval (s)", ymin=0.6, ymax=1.9, yticks=[0.7, 1.2, 1.7])

    myfig.Line(p0, x=[0, 25, 50, 100], y=interbout_interval_as_function_of_coherence_wt_mean, yerr=interbout_interval_as_function_of_coherence_wt_sem, lc="black", zorder=1)
    myfig.Scatter(p0, x=[0, 25, 50, 100], y=interbout_interval_as_function_of_coherence_wt_mean, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

    myfig.Line(p0, x=[0, 25, 50, 100], y=interbout_interval_as_function_of_coherence_mutant_mean, yerr=interbout_interval_as_function_of_coherence_mutant_sem, lc=basecolor, zorder=1)
    myfig.Scatter(p0, x=[0, 25, 50, 100], y=interbout_interval_as_function_of_coherence_mutant_mean, lc=basecolor, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

    ####
    # Correctness over time as function of coherence
    p0 = myfig.Plot(fig, num='b3', xpos=13, ypos=ypos - 1.5, plot_height=2.5, title = "Wildtype",
                                       plot_width=2.2, errorbar_area=False, lw=1,
                                       xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                                       yl="Probability correct (%)", ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])

    for i in range(4):
        c = colors.hsv_to_rgb([0, 0, (i + 2) / 8])

        myfig.Line(p0, x=binned_correctness_wt_mean.loc[i, :].index[1:] - 10, y=binned_correctness_wt_mean.loc[i, :].values[1:][:,0], yerr=binned_correctness_wt_sem.loc[i, :].values[1:][:,0], lc=c, zorder=1)
        myfig.Scatter(p0, x=binned_correctness_wt_mean.loc[i, :].index[1:] - 10, y=binned_correctness_wt_mean.loc[i, :].values[1:][:,0], lc=c, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)


    # Correctness over time as function of coherence
    p0 = myfig.Plot(fig, num='', xpos=16, ypos=ypos - 1.5, plot_height=2.5, title = "Mutant",
                                       plot_width=2.2, errorbar_area=False, lw=1,
                                       xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                                       ymin=44, ymax=91, yticks=[50, 70, 90], yticklabels=[""]*3, vspans=[[0, 10, "#cccccc", 0.6]])


    h, s, v = colors.rgb_to_hsv(colors.to_rgb(basecolor))

    for i in range(4):
        c = colors.hsv_to_rgb([h, s*(4-i)/4, v*(i+2)/6])

        myfig.Line(p0, x=binned_correctness_mutant_mean.loc[i, :].index[1:] - 10, y=binned_correctness_mutant_mean.loc[i, :].values[1:][:,0], yerr=binned_correctness_mutant_sem.loc[i, :].values[1:][:,0], lc=c, zorder=1)
        myfig.Scatter(p0, x=binned_correctness_mutant_mean.loc[i, :].index[1:] - 10, y=binned_correctness_mutant_mean.loc[i, :].values[1:][:,0], lc=c, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

    ####
    # Same direction
    p0 = myfig.Plot(fig, num='b4', xpos=10, ypos=ypos - 4, plot_height=1.25, plot_width=1.3,
                               errorbar_area=False, lw=1,
                               xl="Time since last turn (s)", xmin=-0.15, xmax=1.6, xticks=[0, 0.5, 1, 1.5], hlines=[50],
                               yl="Probability to turn\nin same direction (%)", ymin=48, ymax=71,
                               yticks=[50, 60, 70], vlines=[0])
    #
    myfig.Line(p0, x=binned_same_direction_wt_mean.index, y=binned_same_direction_wt_mean.values, yerr=binned_same_direction_wt_sem.values[:,0], lc="black", zorder=1)
    myfig.Scatter(p0, x=binned_same_direction_wt_mean.index, y=binned_same_direction_wt_mean.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

    myfig.Line(p0, x=binned_same_direction_mutant_mean.index, y=binned_same_direction_mutant_mean.values, yerr=binned_same_direction_mutant_sem.values[:,0], lc=basecolor, zorder=1)
    myfig.Scatter(p0, x=binned_same_direction_mutant_mean.index, y=binned_same_direction_mutant_mean.values, lc=basecolor, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

    ####
    # Turning histograms
    p0 = myfig.Plot(fig, num='b5', xpos=13, ypos=ypos - 4, plot_height=1.5, plot_width=1.8,
                               errorbar_area=True, lw=1,
                               xl="Turn angle (deg)", xmin=-101, xmax=121, xticks=[-100, -50, 0, 50, 100], hlines=[50],
                               yl="Probability density", ymin=-0.01, ymax=0.3,
                               yticks=[], vlines=[0])

    for i in range(4):
        c = colors.hsv_to_rgb([0, 0, (i + 2) / 8])
        myfig.Line(p0, x=histogram_heading_angle_change_wt_mean.loc[i,:].index, y=histogram_heading_angle_change_wt_mean.loc[i,:].values[:,0], yerr=histogram_heading_angle_change_wt_sem.loc[i,:].values[:,0], lc=c, zorder=1)

    # The mutant
    p0 = myfig.Plot(fig, num='', xpos=16, ypos=ypos - 4, plot_height=1.5, plot_width=1.8,
                               errorbar_area=True, lw=1,
                               xl="Turn angle (deg)", xmin=-101, xmax=121, xticks=[-100, -50, 0, 50, 100], hlines=[50],
                               yl="", ymin=-0.01, ymax=0.3,
                               yticks=[], vlines=[0])

    h, s, v = colors.rgb_to_hsv(colors.to_rgb(basecolor))
    for i in range(4):
        c = colors.hsv_to_rgb([h, s * (4 - i) / 4, v * (i + 2) / 6])
        myfig.Line(p0, x=histogram_heading_angle_change_mutant_mean.loc[i,:].index, y=histogram_heading_angle_change_mutant_mean.loc[i,:].values[:,0], yerr=histogram_heading_angle_change_wt_sem.loc[i,:].values[:,0], lc=c, zorder=1)



# Do a bootstrap test
def bootstrap(vals1, vals2):

    combined = np.r_[vals1, vals2]
    ds = []
    for i in range(10000):
        ds.append(np.median(np.random.choice(combined, 12)) - np.median(np.random.choice(combined, 12)))

    ds = np.array(ds)
    d_real = np.abs(np.median(vals1) - np.median(vals2))

    p = (ds > d_real).sum() / len(ds)
    print(p)
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "ns"

    return p, stars

# Show errors over fitting runs
#####
# Display the evolution of error functions over generation

p0 = myfig.Plot(fig, num='b', xpos=2.0, ypos=7, plot_height=0.75, plot_width=1.5,
                               lw=1, pc='white', errorbar_area=False,
                               xl="Generation", xmin=-1, xmax=4.5, xticks=[0, 1, 2.5, 3.5], xticklabels=["Initial", "Final", "Initial", "Final"], xticklabels_rotation=45,
                               yl="Minimal error\nin population", ymin=-0.01, ymax=0.51, yticks=[0, 0.25, 0.5], hlines=[0])

# Only show the consensus error
for i in range(12):
    myfig.Scatter(p0, x=[0], y=df_minimal_error_gen0_wt.query("error_i == 5").values[i], pt='o', lw=0.5, ps=9.8, pc='white', lc='black', zorder=2, alpha=0.5)
    myfig.Scatter(p0, x=[1], y=df_minimal_error_gen79_wt.query("error_i == 5").values[i], pt='o', lw=0.5, ps=9.8, pc='white', lc='black', zorder=2, alpha=0.5)
    myfig.Line(p0, x=[0, 1], y=[df_minimal_error_gen0_wt.query("error_i == 5").values[i],
                                df_minimal_error_gen79_wt.query("error_i == 5").values[i]], lw=0.5, lc='black', zorder=1)

    myfig.Scatter(p0, x=[2.5], y=df_minimal_error_gen0_mutant.query("error_i == 5").values[i], pt='o', lw=0.5, ps=9.8, pc='white', lc=basecolor, zorder=2, alpha=0.5)
    myfig.Scatter(p0, x=[3.5], y=df_minimal_error_gen79_mutant.query("error_i == 5").values[i], pt='o', lw=0.5, ps=9.8, pc='white', lc=basecolor, zorder=2, alpha=0.5)
    myfig.Line(p0, x=[2.5, 3.5], y=[df_minimal_error_gen0_mutant.query("error_i == 5").values[i],
                                    df_minimal_error_gen79_mutant.query("error_i == 5").values[i]], lw=0.5, lc=basecolor, zorder=1)

# Tau
p0 = myfig.Plot(fig, num='', xpos=5.0, ypos=7, plot_height=0.75, plot_width=1.25, title="Tau",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=4.1, xticks=[0, 2, 4],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=["Wildtype", "Mutant"])

myfig.Scatter(p0, x=df_estimated_parameters_wt['tau'].values, y=[0]*12, lc='black', pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_wt['tau'].median()], y=[0], lc='black', pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

myfig.Scatter(p0, x=df_estimated_parameters_mutant['tau'].values, y=[1]*12, lc=basecolor, pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_mutant['tau'].median()], y=[1], lc=basecolor, pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

p, stars = bootstrap(df_estimated_parameters_wt['tau'].values, df_estimated_parameters_mutant['tau'].values)
myfig.Line(p0, x=[3, 3], y=[0,1], lc='black', lw=0.75)
myfig.Text(p0, x=3.2, y=0.5, text=stars)

# Sigma
p0 = myfig.Plot(fig, num='', xpos=7, ypos=7, plot_height=0.75, plot_width=1.25, title="sigma",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=30.1, xticks=[0, 15, 30],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=["", ""])

myfig.Scatter(p0, x=df_estimated_parameters_wt['noise_sigma'].values, y=[0]*12, lc='black', pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_wt['noise_sigma'].median()], y=[0], lc='black', pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

myfig.Scatter(p0, x=df_estimated_parameters_mutant['noise_sigma'].values, y=[1]*12, lc=basecolor, pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_mutant['noise_sigma'].median()], y=[1], lc=basecolor, pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

p, stars = bootstrap(df_estimated_parameters_wt['noise_sigma'].values, df_estimated_parameters_mutant['noise_sigma'].values)
myfig.Line(p0, x=[15, 15], y=[0,1], lc='black', lw=0.75)
myfig.Text(p0, x=15.5, y=0.5, text=stars)


# T
p0 = myfig.Plot(fig, num='', xpos=9, ypos=7, plot_height=0.75, plot_width=1.25, title="T",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value", xmin=-0.1, xmax=2.1, xticks=[0, 1, 2],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=["", ""])

myfig.Scatter(p0, x=df_estimated_parameters_wt['T'].values, y=[0]*12, lc='black', pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_wt['T'].median()], y=[0], lc='black', pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

myfig.Scatter(p0, x=df_estimated_parameters_mutant['T'].values, y=[1]*12, lc=basecolor, pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_mutant['T'].median()], y=[1], lc=basecolor, pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

p, stars = bootstrap(df_estimated_parameters_wt['T'].values,
                     df_estimated_parameters_mutant['T'].values)
myfig.Line(p0, x=[1.5, 1.5], y=[0,1], lc='black', lw=0.75)
myfig.Text(p0, x=1.7, y=0.5, text=stars)

# p_below
p0 = myfig.Plot(fig, num='', xpos=11.0, ypos=7, plot_height=0.75, plot_width=1.25, title="p_below",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value",  xmin=-0.44, xmax=2.1, xticks=[0, 1, 2],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=["", ""])

myfig.Scatter(p0, x=df_estimated_parameters_wt['bout_clock_probability_below_threshold'].values*100, y=[0]*12, lc='black', pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_wt['bout_clock_probability_below_threshold'].median()*100], y=[0], lc='black', pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

myfig.Scatter(p0, x=df_estimated_parameters_mutant['bout_clock_probability_below_threshold'].values*100, y=[1]*12, lc=basecolor, pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_mutant['bout_clock_probability_below_threshold'].median()*100], y=[1], lc=basecolor, pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

p, stars = bootstrap(df_estimated_parameters_wt['bout_clock_probability_below_threshold'].values*100,
                     df_estimated_parameters_mutant['bout_clock_probability_below_threshold'].values*100)
myfig.Line(p0, x=[1.7, 1.7], y=[0,1], lc='black', lw=0.75)
myfig.Text(p0, x=1.75, y=0.5, text=stars)

# p_above
p0 = myfig.Plot(fig, num='', xpos=13.0, ypos=7, plot_height=0.75, plot_width=1.25, title="p_above",
            lw=1, pc='white', errorbar_area=False,
            xl="Estimated value",  xmin=-0.44, xmax=6.1, xticks=[0, 2, 4, 6],
            yl="", ymin=-0.6, ymax=1.6, yticks=[0, 1], yticklabels=["", ""])

myfig.Scatter(p0, x=df_estimated_parameters_wt['bout_clock_probability_above_threshold'].values*100, y=[0]*12, lc='black', pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_wt['bout_clock_probability_above_threshold'].median()*100], y=[0], lc='black', pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

myfig.Scatter(p0, x=df_estimated_parameters_mutant['bout_clock_probability_above_threshold'].values*100, y=[1]*12, lc=basecolor, pt='o',
                      lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)
myfig.Scatter(p0, x=[df_estimated_parameters_mutant['bout_clock_probability_above_threshold'].median()*100], y=[1], lc=basecolor, pt='o',
                      lw=1, ps=10, pc='white', zorder=2)

p, stars = bootstrap(df_estimated_parameters_wt['bout_clock_probability_above_threshold'].values*100,
                     df_estimated_parameters_mutant['bout_clock_probability_above_threshold'].values*100)
myfig.Line(p0, x=[3, 3], y=[0,1], lc='black', lw=0.75)
myfig.Text(p0, x=3.5, y=0.5, text=stars)

fig.savepdf(target_path / f"raw_figure2_{experiment}", open_pdf=True)
