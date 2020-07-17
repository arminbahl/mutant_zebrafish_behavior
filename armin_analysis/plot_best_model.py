from pathlib import Path
import numpy as np
from fit_integrator_model import get_model_result, get_target_result
import my_figure as myfig
import matplotlib
import pandas as pd

#root_path = Path("/Users/arminbahl/Desktop/preprocessed data/4 fish setups free swimming behavior/dot_motion_coherence8_2")
root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/scn1lab_sa16474")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/scn1lab_NIBR")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/immp2l_summer")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/immp2l_NIBR")

found_parameters = []

for genotype in ["wt", "wt", "wt"]:

    target_correctness_as_function_of_coherence, \
    target_inter_bout_interval_as_function_of_coherence, \
    target_binned_correctness, \
    target_binned_same_direction, \
    target_binned_features_heading_angle_change_histograms,\
    target_binned_features_inter_bout_interval_histograms = get_target_result(root_path, genotype)

    print(target_correctness_as_function_of_coherence)

    for repeat in range(1):
        try:
            X = np.load(root_path / f"leaky_integrator_model2_X_{genotype}_{repeat}.npy")
            F = np.load(root_path / f"leaky_integrator_model2_F_{genotype}_{repeat}.npy")
        except:
            continue

        fig = myfig.Figure(title=f"Model fits to {genotype}", fig_width=26)

        p1_experiment = myfig.Plot(fig, num='a', xpos=1.5, ypos=22, plot_height=1.25, plot_width=1.25, title="Experiment",
                                   lw=1, pc='white', errorbar_area=False,
                                   xl="Coherence (%)", xmin=-15, xmax=115, hlines=[50], xticks=[0, 25, 50, 100],
                                   yl="Target function 0\n\nProbability\ncorrect (%)", ymin=44, ymax=91, yticks=[50, 70, 90])

        p1_model = [myfig.Plot(fig, num='', xpos=1.5 + 3*(optimized_i + 1), ypos=22, plot_height=1.25,
                               title=f"Best for function {optimized_i}" if optimized_i < 6 else "Preferred function weight",
                               plot_width=1.25, lw=1, pc='white', errorbar_area=False,
                               xl="Coherence (%)", xmin=-15, xmax=115, hlines=[50], xticks=[0, 25, 50, 100],
                               ymin=44, ymax=91, yticks=[50, 70, 90])
                    for optimized_i in range(7)]

        ##########
        p2_experiment = myfig.Plot(fig, num='b', xpos=1.5, ypos=19, plot_height=1.25, plot_width=1.25,
                                   lw=1, pc='white', errorbar_area=False,
                                   xl="Coherence (%)", xmin=-15, xmax=115, xticks=[0, 25, 50, 100], hlines=[50],
                                   yl="Target function 1\n\nInterbout\ninterval (s)", ymin=0.45, ymax=1.05, yticks=[0.5, 0.75, 1.0])

        p2_model = [myfig.Plot(fig, num='', xpos=1.5 + 3*(optimized_i + 1), ypos=19, plot_height=1.25, plot_width=1.25,
                               lw=1, pc='white', errorbar_area=False,
                               xl="Coherence (%)", xmin=-15, xmax=115, xticks=[0, 25, 50, 100], hlines=[50],
                               ymin=0.45, ymax=1.05, yticks=[0.5, 0.75, 1.0])
                    for optimized_i in range(7)]

        ##########
        p3_experiment = myfig.Plot(fig, num='c', xpos=1.5, ypos=15, plot_height=2.75,
                                   plot_width=2.2, errorbar_area=False, lw=1,
                                   xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                                   yl="Target function 2\n\nProbability correct (%)", ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])

        p3_model = [myfig.Plot(fig, num='', xpos=1.5 + 3*(optimized_i + 1), ypos=15, plot_height=2.75, plot_width=2.2,
                               errorbar_area=False, lw=1,
                               xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                               ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])
                    for optimized_i in range(7)]

        ##########
        p4_experiment = myfig.Plot(fig, num='d', xpos=1.5, ypos=11, plot_height=2.75, plot_width=2.2,
                                   errorbar_area=False, lw=1,
                                   xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                                   yl="Target function 3\n\nProbability correct (%)", ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])

        p4_model = [myfig.Plot(fig, num='', xpos=1.5 + 3*(optimized_i + 1), ypos=11, plot_height=2.75, plot_width=2.2,
                               errorbar_area=False, lw=1,
                               xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                               ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])
                    for optimized_i in range(7)]

        ##########
        p5_experiment = myfig.Plot(fig, num='d', xpos=1.5, ypos=7, plot_height=2.75, plot_width=2.2,
                                   errorbar_area=False, lw=1,
                                   xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                                   yl="Target function 4\n\nProbability correct (%)", ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])

        p5_model = [myfig.Plot(fig, num='', xpos=1.5 + 3*(optimized_i + 1), ypos=7, plot_height=2.75, plot_width=2.2,
                               errorbar_area=False, lw=1,
                               xl="Time (s)", xmin=-2, xmax=14, xticks=[0, 5, 10], hlines=[50],
                               ymin=44, ymax=91, yticks=[50, 70, 90], vspans=[[0, 10, "#cccccc", 0.6]])
                    for optimized_i in range(7)]

        ##########
        p6_experiment = myfig.Plot(fig, num='d', xpos=1.5, ypos=3, plot_height=2.75, plot_width=1.3,
                                   errorbar_area=False, lw=1,
                                   xl="Time since last turn (s)", xmin=-0.15, xmax=1.6, xticks=[0, 0.5, 1, 1.5], hlines=[50],
                                   yl="Target function 5\n\nProbability to turn\nin same direction (%)", ymin=44, ymax=91, yticks=[50, 70, 90], vlines=[0])

        p6_model = [myfig.Plot(fig, num='', xpos=1.5 + 3*(optimized_i + 1), ypos=3, plot_height=2.75, plot_width=1.3,
                               errorbar_area=False, lw=1,
                               xl="Time since last turn (s)", xmin=-0.15, xmax=1.6, xticks=[0, 0.5, 1, 1.5], hlines=[50],
                               ymin=44, ymax=91, yticks=[50, 70, 90], vlines=[0])
                    for optimized_i in range(7)]

        # Show the target experimental data
        myfig.Line(p1_experiment, x=[0, 25, 50, 100], y=target_correctness_as_function_of_coherence.values, lc="black", zorder=1)
        myfig.Scatter(p1_experiment, x=[0, 25, 50, 100], y=target_correctness_as_function_of_coherence.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        myfig.Line(p2_experiment, x=[0, 25, 50, 100], y=target_inter_bout_interval_as_function_of_coherence.values, lc="black", zorder=1)
        myfig.Scatter(p2_experiment, x=[0, 25, 50, 100], y=target_inter_bout_interval_as_function_of_coherence.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        myfig.Line(p3_experiment, target_binned_correctness.loc[1, :].index - 10, target_binned_correctness.loc[1, :].values, lc="black", zorder=1)
        myfig.Scatter(p3_experiment, x=target_binned_correctness.loc[1, :].index - 10, y=target_binned_correctness.loc[1, :].values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        myfig.Line(p4_experiment, target_binned_correctness.loc[2, :].index - 10, target_binned_correctness.loc[2, :].values, lc="black", zorder=1)
        myfig.Scatter(p4_experiment, x=target_binned_correctness.loc[2, :].index - 10, y=target_binned_correctness.loc[2, :].values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        myfig.Line(p5_experiment, target_binned_correctness.loc[3, :].index - 10, target_binned_correctness.loc[3, :].values, lc="black", zorder=1)
        myfig.Scatter(p5_experiment, x=target_binned_correctness.loc[3, :].index - 10, y=target_binned_correctness.loc[3, :].values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        myfig.Line(p6_experiment, target_binned_same_direction.index, target_binned_same_direction.values, lc="black", zorder=1)
        myfig.Scatter(p6_experiment, x=target_binned_same_direction.index, y=target_binned_same_direction.values, lc='black', pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

        norm = matplotlib.colors.Normalize(vmin=-10, vmax=50)
        cmap = matplotlib.cm.get_cmap('Reds')

        for optimized_i in range(7):
            print(repeat, optimized_i)
            for gen in [49]:

                if optimized_i < 6:
                    i = np.argmin(F[gen, :, optimized_i])
                else:
                    i = np.argmin(F[gen, :, 1]*2500 + F[gen, :, 3])
                #print(X[gen, i])
                #cfg
                correctness_as_function_of_coherence, \
                inter_bout_interval_as_function_of_coherence, \
                binned_correctness, \
                binned_same_direction, \
                binned_features_heading_angle_change_histograms,\
                binned_features_inter_bout_interval_histograms = get_model_result(X[gen, i])

                lc = cmap(norm(gen))
                myfig.Line(p1_model[optimized_i], x=[0, 25, 50, 100], y=correctness_as_function_of_coherence.values, lc=lc, zorder=1)
                myfig.Scatter(p1_model[optimized_i], x=[0, 25, 50, 100], y=correctness_as_function_of_coherence.values, lc=lc, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

                myfig.Line(p2_model[optimized_i], x=[0, 25, 50, 100], y=inter_bout_interval_as_function_of_coherence.values, lc=lc, zorder=1)
                myfig.Scatter(p2_model[optimized_i], x=[0, 25, 50, 100], y=inter_bout_interval_as_function_of_coherence.values, lc=lc, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

                myfig.Line(p3_model[optimized_i], x=binned_correctness.loc[1, :].index - 10, y=binned_correctness.loc[1, :].values, lc=lc, zorder=1)
                myfig.Scatter(p3_model[optimized_i], x=binned_correctness.loc[1, :].index - 10, y=binned_correctness.loc[1, :].values, lc=lc, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

                myfig.Line(p4_model[optimized_i], x=binned_correctness.loc[2, :].index - 10, y=binned_correctness.loc[2, :].values, lc=lc, zorder=1)
                myfig.Scatter(p4_model[optimized_i], x=binned_correctness.loc[2, :].index - 10, y=binned_correctness.loc[2, :].values, lc=lc, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

                myfig.Line(p5_model[optimized_i], x=binned_correctness.loc[3, :].index - 10, y=binned_correctness.loc[3, :].values, lc=lc, zorder=1)
                myfig.Scatter(p5_model[optimized_i], x=binned_correctness.loc[3, :].index - 10, y=binned_correctness.loc[3, :].values, lc=lc, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

                myfig.Line(p6_model[optimized_i], x=binned_same_direction.index, y=binned_same_direction.values, lc=lc, zorder=1)
                myfig.Scatter(p6_model[optimized_i], x=binned_same_direction.index, y=binned_same_direction.values, lc=lc, pt='o', lw=0.5, ps=9.8, pc='white', zorder=2)

                found_parameters.append([genotype, repeat, optimized_i, gen] + list(X[gen, i, :]))

            parameters = f"tau = {X[gen, i, 0]:.1f}\n" \
                         f"sigma = {X[gen, i, 1]:.1f}\n" \
                         f"T = {X[gen, i, 2]:.1f}\n" \
                         f"p_below = {X[gen, i, 3]:.3f}\n" \
                         f"p_above = {X[gen, i, 4]:.3f}"

            fig.add_text(1.5 + 3*(optimized_i + 1) + 1.25/2, 25, parameters)

        fig.savepdf(root_path / f"optimized_model_{genotype}_{repeat}", open_pdf=True)
#
# Save the found optimal parameters
# df = pd.DataFrame(found_parameters,
#                   columns=["genotype",
#                            "repeat",
#                            "optimized_i",
#                            "gen",
#                            "tau",
#                            "sigma",
#                            "T",
#                            "p_below",
#                            "p_above"]).astype(dtype={"repeat": "int64", "optimized_i": "int64", "gen": "int64"}, copy=False)
#
# df.set_index(["genotype", 'repeat', 'gen', "optimized_i"], inplace=True)
# df.sort_index(inplace=True)
#
# df.to_hdf(root_path / "found_parameters.h5", key="parameters", complevel=9)