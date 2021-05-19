import pylab as pl
from get_fish_info import get_fish_info
from fit_integrator_model import get_model_result, get_target_result
import numpy as np
from pathlib import Path
import gmm_model_fit
import pandas as pd
from pymoo.factory import get_problem, get_visualization, get_decomposition
from sklearn.manifold import TSNE
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated

root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/mutant_behavior_data/dot_motion_coherence/scn1lab_NIBR_20200708")

errornames = ["Distance 0 (Correctness as\nfunction of coherence)",
              "Distance 1 (Inter-bout interval\nas function of coherence)",
              "Distance 2 (Binned correctness\nat 25, 50, and 100 %)",
              "Distance 3 (Binned same direction)",
              "Distance 4 (Turn angle\ndistribution peaks)"]

#errornames = ["Mixed"]
genotype = "wt"
repeat = 4
X = np.load(root_path / f"leaky_integrator_model2_X_{genotype}_{repeat}.npy")
F = np.load(root_path / f"leaky_integrator_model2_F_{genotype}_{repeat}.npy")

#sorter = NonDominatedSorting()
#k = sorter.do(F[-1, :, :])

# F = np.random.random((1000, 5))
# # use fast non-dominated sorting
# res = NonDominatedSorting().do(F)
#
# pl.plot(F[:,0], F[:,1],'o')
# for i in range(len(res)):
#     pl.plot(F[res[i]][:,0], F[res[i]][:,1],'.')
# pl.show()
# sadf
import matplotlib
print(F.shape)
import my_figure as myfig

norm = matplotlib.colors.Normalize(vmin=0, vmax=79)
cmap = matplotlib.cm.get_cmap('seismic')

fig = myfig.Figure(title="Pareto fronts")

y_pos = 22
for error_index0, error_index1, xmin, xmax, xticks, ymin, ymax, yticks in [[0, 1, -10, 1010, [0, 500, 1000], -0.1, 5.1, [0, 2.5, 5]],
                                                                           [1, 2, -0.01, 0.6, [0, 0.25, 0.5], -10, 5010, [0, 2500, 5000]],
                                                                           [2, 3, -10, 5010, [0, 2500, 5000], -5, 205, [0, 100, 200]],
                                                                           [3, 4, -10, 510, [0, 250, 500], -0.01, 0.51, [0, 0.25, 0.5]],
                                                                           ]:

        p0 = myfig.Plot(fig, num='', xpos=1.5, ypos=y_pos, plot_height=2.5, plot_width=2.5,
                        lw=1, pc='white', errorbar_area=False,
                        xl=errornames[error_index0], xmin=xmin, xmax=xmax, xticks=xticks,
                        yl=errornames[error_index1], ymin=ymin, ymax=ymax,yticks=yticks, hlines=[0], vlines=[0])


        print(F[0][:, error_index0].max())
        print(F[0][:, error_index1].max())
        res = NonDominatedSorting().do(F[0])
        x = F[0][res[0]][:, error_index0]
        y = F[0][res[0]][:, error_index1]
        ind = np.where((x < xmax) & (y < ymax))
        x = x[ind]
        y = y[ind]

        myfig.Scatter(p0, x,y, pt='.', ps=8, lw=0, pc='blue',alpha=0.8)

        res = NonDominatedSorting().do(F[-1])
        x = F[-1][res[0]][:, error_index0]
        y = F[-1][res[0]][:, error_index1]
        ind = np.where((x < xmax) & (y < ymax))
        x = x[ind]
        y = y[ind]

        myfig.Scatter(p0, x, y, pt='.', ps=8, lw=0, pc='red',alpha=0.3)

        y_pos-= 4

tsne = TSNE(n_components=2, random_state=1)

res_initial = NonDominatedSorting().do(F[0])
res_end = NonDominatedSorting().do(F[-1])

j_initial = np.zeros_like(F[0, :,0])
j_initial[res_initial[0]] = 1

j_end = np.zeros_like(F[-1, :, 0])
j_end[res_end[0]] = 1

tsne_initial = tsne.fit_transform(F[0])
tsne_end = tsne.fit_transform(F[-1])

p0 = myfig.Plot(fig, num='', xpos=10, ypos=22, plot_height=2.5, plot_width=2.5,
                lw=1, pc='white', errorbar_area=False,
                xl="tSNE dim 1", xmin=-40, xmax=40, xticks=[-40, -20, 0,20,40],
                yl="tSNE dim 2", ymin=-40, ymax=40, yticks=[-40, -20, 0,20,40], hlines=[0], vlines=[0])

p1 = myfig.Plot(fig, num='', xpos=10, ypos=15, plot_height=2.5, plot_width=2.5,
                lw=1, pc='white', errorbar_area=False,
                xl="tSNE dim 1", xmin=-40, xmax=40, xticks=[-40, -20, 0, 20, 40],
                yl="tSNE dim 2", ymin=-40, ymax=40, yticks=[-40, -20, 0, 20, 40], hlines=[0], vlines=[0])

# print(j_initial.shape)
# print(tsne_initial.shape)
# sdf
for i in range(j_initial.shape[0]):
    #print(tsne_initial[i, 0].max(), tsne_initial[i, 0].min())
    if j_initial[i] == 0:
        myfig.Scatter(p0, tsne_initial[i, 0], tsne_initial[i, 1], pt='.', ps=8, lw=0, pc='gray',alpha=0.3)
    else:
        myfig.Scatter(p0, tsne_initial[i, 0], tsne_initial[i, 1], pt='.', ps=8, lw=0, pc='blue',alpha=0.8)

for i in range(j_end.shape[0]):
    if j_end[i] == 0:
        myfig.Scatter(p1, tsne_end[i, 0], tsne_end[i, 1], pt='.', ps=8, lw=0, pc='gray',alpha=0.3)
    else:
        myfig.Scatter(p1, tsne_end[i, 0], tsne_end[i, 1], pt='.', ps=8, lw=0, pc='red',alpha=0.3)


fig.savepdf(root_path / f"pareto_tests", open_pdf=True)
asfd


print(X.shape)
sdf
pl.plot(F[-1, :, 0], F[-1, :, 1], '.')
pl.plot(F[0, :, 0], F[0, :, 1], '.')
pl.xlim(0, 0.06)
pl.ylim(0, 0.02)

pl.figure()

pl.show()

print(X.shape, F.shape)
sdf

#
#
# for i in range(7):
#     F[-1, :, i] = F[-1, :, i] / np.max(F[-1, :, i])
# print(F.shape)
#
# i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] + F[-1, :, 2] + F[-1, :, 3] + F[-1, :, 4] + F[-1, :, 5] + F[-1, :, 6])
# print(F[-1, i6, 0])
# dd
#get_decomposition("asf").do(F[-1], [1, 1, 1, 1, 1, 1, 1]).argmin()
#print(I)
#sdfsdf
#X = np.load(root_path / f"leaky_integrator_model2_X_{genotype}_{repeat}_single_error.npy")
#F = np.load(root_path / f"leaky_integrator_model2_F_{genotype}_{repeat}_single_error.npy")

# from pymoo.factory import get_decision_making, get_reference_directions
#
# ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)
# F = get_problem("dtlz1").pareto_front(ref_dirs)
#
# weights = np.array([10.25, 10.25, 0.25, 0.25])
# a, pseudo_weights = get_decision_making("pseudo-weights", weights).do(F, return_pseudo_weights=True)
# pl.plot(F[:, 0], F[:,1], 'o')
# pl.plot(F[a, 0], F[a,1], 'o')
# pl.show()
#
# print(a, pseudo_weights, F.shape)
# ghj
from pymoo.factory import get_decision_making, get_reference_directions


#weights = [1000, 1000, 1000, 0, 0, 0, 0]
#a, pseudo_weights = get_decision_making("pseudo-weights", weights).do(F[-1], return_pseudo_weights=True)
#print(pseudo_weights[0])
#print(a, pseudo_weights)
#dfg
for i in range(5):
    #pl.hist(F[-1, :, i])
    #pl.show()

    #print(np.percentile(F[-1, :, i], 75))
    #print(np.max(F[-1, :, i]) - np.min(F[-1, :, i]))
    F[-1, :, i] = F[-1, :, i] / np.percentile(F[-1, :, i], 75)
# print(F.shape)
#

#i6 = a

#i1 = np.argmin(F[-1, :, 0])
# i2 = np.argmin(F[-1, :, 1])
# i3 = np.argmin(F[-1, :, 0] + F[-1, :, 1]*500)
# i4 = np.argmin(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3])
# i5 = np.argmin(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3] + F[-1, :, 5]*0.25)
# #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3] + F[-1, :, 5]*0.25 + F[-1, :, 6]*5800)
# #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] * 2500 + F[-1, :, 3] * 5 + F[-1, :, 5] * 0.5 + F[-1, :, 6] * 6800)
# i6 = np.argmin(F[-1, :, 0]*500 + F[-1, :, 1]*2500 + F[-1, :, 3]*50 + F[-1, :, 5]*0.5 + F[-1, :, 6]*4500)

i6 = np.argmin(F[-1, :, 0] + 3*F[-1, :, 1] + F[-1, :, 2] + F[-1, :, 3] + F[-1, :, 4])


#from pymoo.factory import get_decision_making
#dm = get_decision_making("high-tradeoff")

#I = dm.do(pf)

# print(F.shape)
# np.set_printoptions(precision=4, suppress=True)
# print((X[-1, i]))
# #gdfgh
# for error_i in range(len(errornames)):
#     pl.figure()
#     pl.title(errornames[error_i])
#     bp = pl.boxplot(F[:, :, error_i].T, whis=[5, 95], showfliers=False, medianprops=medianprops)
#     for gen in range(50):
#         sc = pl.scatter([gen+1], [F[gen, :, error_i].min()], s=5, marker='.', c='firebrick')
#     pl.yscale("log")
#     pl.xlabel("Generation")
#     pl.ylabel("Log Error")
# pl.show()
# dd
#
# pl.figure()
# pl.title("Compromise between all error functions")
# #error = F[:, :, 0] + F[:, :, 1]*500 + F[:, :, 3] + F[:, :, 5]*0.25 + F[:, :, 6]*500
# error = F[:, :, 0] + F[:, :, 1]*2500 + F[:, :, 3]*5 + F[:, :, 5]*0.5 + F[:, :, 6]*1500
#
# bp = pl.boxplot(error.T, whis=[5, 95], showfliers=False, medianprops=medianprops)
# for gen in range(50):
#     sc = pl.scatter([gen + 1], [error[gen].min()], s=10, marker='.', c='firebrick')
# pl.yscale("log")
# pl.xlabel("Generation")
# pl.ylabel("Log Error")
# pl.show()

# pl.figure()
# pl.scatter(F[-1, :, 0], F[-1, :, 1], s=10, marker='.', c='C0', label='Individual')
# pl.scatter(F[-1, i1, 0], F[-1, i1, 1], s=15, marker='o', c='C1', label="Best for 'Correctness as function of coherence'")
# pl.scatter(F[-1, i2, 0], F[-1, i2, 1], s=15, marker='o', c='C2', label="Best for 'Inter-bout interval as function of coherence'")
# pl.scatter(F[-1, i3, 0], F[-1, i3, 1], s=15, marker='o', c='C3', label="Compromise")
# pl.legend()
# pl.xlabel(errornames[0])
# pl.ylabel(errornames[1])
#
#
# pl.figure()
# pl.scatter(F[-1, :, 0] + F[-1, :, 1]*500, F[-1, :, 3], s=10, marker='.', c='C0', label='Individual')
# pl.scatter(F[-1, i1, 0] + F[-1, i1, 1]*500, F[-1, i1, 3], s=15, marker='o', c='C1', label="Best for 'Correctness as function of coherence'")
# pl.scatter(F[-1, i2, 0] + F[-1, i2, 1]*500, F[-1, i2, 3], s=15, marker='o', c='C2', label="Best for 'Inter-bout interval as function of coherence'")
# pl.scatter(F[-1, i3, 0] + F[-1, i3, 1]*500, F[-1, i3, 3], s=15, marker='o', c='C3', label="Compromise between 1 and 2")
# pl.scatter(F[-1, i4, 0] + F[-1, i4, 1]*500, F[-1, i4, 3], s=15, marker='o', c='C4', label="Compromise between all")
# pl.legend()
# pl.xlabel("Compromise between 1 and 2")
# pl.ylabel(errornames[3])
#
# pl.figure()
# pl.scatter(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3], F[-1, :, 5], s=10, marker='.', c='C0', label='Individual')
# pl.scatter(F[-1, i1, 0] + F[-1, i1, 1]*500 + F[-1, i1, 3], F[-1, i1, 5], s=15, marker='o', c='C1', label="Best for 'Correctness as function of coherence'")
# pl.scatter(F[-1, i2, 0] + F[-1, i2, 1]*500 + F[-1, i2, 3], F[-1, i2, 5], s=15, marker='o', c='C2', label="Best for 'Inter-bout interval as function of coherence'")
# pl.scatter(F[-1, i3, 0] + F[-1, i3, 1]*500 + F[-1, i3, 3], F[-1, i3, 5], s=15, marker='o', c='C3', label="Compromise between 1 and 2")
# pl.scatter(F[-1, i4, 0] + F[-1, i4, 1]*500 + F[-1, i4, 3], F[-1, i4, 5], s=15, marker='o', c='C4', label="Compromise between 1, 2, and 3")
# pl.scatter(F[-1, i5, 0] + F[-1, i5, 1]*500 + F[-1, i5, 3], F[-1, i5, 5], s=15, marker='o', c='C5', label="Compromise between all")
# pl.legend()
# pl.xlabel("Compromise between 1, 2, and 3")
# pl.ylabel(errornames[5])
#
#
# pl.figure()
# pl.scatter(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3] + F[-1, :, 5]*0.25, F[-1, :, 6], s=10, marker='.', c='C0', label='Individual')
# pl.scatter(F[-1, i1, 0] + F[-1, i1, 1]*500 + F[-1, i1, 3] + F[-1, i1, 5]*0.25, F[-1, i1, 6], s=15, marker='o', c='C1', label="Best for 'Correctness as function of coherence'")
# pl.scatter(F[-1, i2, 0] + F[-1, i2, 1]*500 + F[-1, i2, 3] + F[-1, i2, 5]*0.25, F[-1, i2, 6], s=15, marker='o', c='C2', label="Best for 'Inter-bout interval as function of coherence'")
# pl.scatter(F[-1, i3, 0] + F[-1, i3, 1]*500 + F[-1, i3, 3] + F[-1, i3, 5]*0.25, F[-1, i3, 6], s=15, marker='o', c='C3', label="Compromise between 1 and 2")
# pl.scatter(F[-1, i4, 0] + F[-1, i4, 1]*500 + F[-1, i4, 3] + F[-1, i4, 5]*0.25, F[-1, i4, 6], s=15, marker='o', c='C4', label="Compromise between 1, 2, and 3")
# pl.scatter(F[-1, i5, 0] + F[-1, i5, 1]*500 + F[-1, i5, 3] + F[-1, i5, 5]*0.25, F[-1, i5, 6], s=15, marker='o', c='C5', label="Compromise between 1, 2, 3, and 4")
# pl.scatter(F[-1, i6, 0] + F[-1, i6, 1]*500 + F[-1, i6, 3] + F[-1, i6, 5]*0.25, F[-1, i6, 6], s=15, marker='o', c='C6', label="Compromise between all")
# pl.legend()
# pl.xlabel("Compromise between 1, 2, 3, and 4")
# pl.ylabel(errornames[6])
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i1])
# fig.suptitle("Best for 'Correctness as function of coherence'")
# pl.subplot(211)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C1')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(212)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C1')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i2])
# fig.suptitle("Best for 'Inter-bout interval as function of coherence'")
# pl.subplot(211)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C2')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(212)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C2')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i3])
# fig.suptitle("Compromise between 'Correctness and inter-bout interval as function of coherence'")
# pl.subplot(211)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C3')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(212)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C3')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i3])
# fig.suptitle("Compromise between 'Correctness and inter-bout interval as function of coherence'")
# pl.subplot(221)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C3')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(222)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C3')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
# pl.subplot(223)
# for i in range(4):
#     pl.plot(target_df_binned_correctness.loc[i, :].droplevel("stim"), 'o-', color='black')
#     pl.plot(model_df_binned_correctness.loc[i, :].droplevel("stim"), 'o--', color='C3')
# pl.xlabel("Correctness (%)")
# pl.ylabel("Time (s)")
#
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i4])
# fig.suptitle("Compromise between all three error functions")
# pl.subplot(221)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C4')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(222)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C4')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
# pl.subplot(223)
# for i in range(4):
#     pl.plot(target_df_binned_correctness.loc[i, :].droplevel("stim"), 'o-', color='black')
#     pl.plot(model_df_binned_correctness.loc[i, :].droplevel("stim"), 'o--', color='C4')
# pl.xlabel("Correctness (%)")
# pl.ylabel("Time (s)")
#
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i5])
# fig.suptitle("Compromise between all four error functions")
# pl.subplot(221)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C5')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(222)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C5')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
# pl.subplot(223)
# for i in range(4):
#     pl.plot(target_df_binned_correctness.loc[i, :].droplevel("stim"), 'o-', color='black')
#     pl.plot(model_df_binned_correctness.loc[i, :].droplevel("stim"), 'o--', color='C5')
# pl.xlabel("Correctness (%)")
# pl.ylabel("Time (s)")
# pl.subplot(224)
# pl.plot(target_df_binned_same_direction, 'o-', color='black')
# pl.plot(model_df_binned_same_direction, 'o--', color='C5')
# pl.xlabel("Time since last bout (s)")
# pl.ylabel("Correctness (%)")

fig = pl.figure()
model_df_correctness_as_function_of_coherence, \
model_df_inter_bout_interval_as_function_of_coherence, \
model_df_binned_correctness, \
model_df_binned_same_direction, \
model_df_binned_features_heading_angle_change_histograms, \
model_df_binned_features_inter_bout_interval_histograms, \
model_df_gmm_fitting_results = get_model_result(X[-1, i6])
fig.suptitle("Compromise between all five error functions")
pl.subplot(231)
pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C6')
pl.xlabel("Coherence (%)")
pl.ylabel("Correctness (%)")
pl.subplot(232)
pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C6')
pl.xlabel("Coherence (%)")
pl.ylabel("Inter-bout interval (s)")
pl.subplot(233)
for i in range(4):
    pl.plot(target_df_binned_correctness.loc[i, :].droplevel("stim"), 'o-', color='black')
    pl.plot(model_df_binned_correctness.loc[i, :].droplevel("stim"), 'o--', color='C6')
pl.xlabel("Time (s)")
pl.ylabel("Correctness (%)")
pl.subplot(234)
pl.plot(target_df_binned_same_direction, 'o-', color='black')
pl.plot(model_df_binned_same_direction, 'o--', color='C6')
pl.xlabel("Time since last bout (s)")
pl.ylabel("Correctness (%)")
# pl.subplot(235)
# pl.plot(target_df_gmm_fitting_results.index*25, target_df_gmm_fitting_results["w_left"].values, '-o', color='black', label='s_left')
# pl.plot(target_df_gmm_fitting_results.index*25, target_df_gmm_fitting_results["w_center"].values, '-o', color='black', label='s_center')
# pl.plot(target_df_gmm_fitting_results.index*25, target_df_gmm_fitting_results["w_right"].values, '-o', color='black', label='s_right')
#
# pl.plot(model_df_gmm_fitting_results.index*25, model_df_gmm_fitting_results["w_left"].values, '--o', color='C6', label='s_left')
# pl.plot(model_df_gmm_fitting_results.index*25, model_df_gmm_fitting_results["w_center"].values, '--o', color='C6', label='s_center')
# pl.plot(model_df_gmm_fitting_results.index*25, model_df_gmm_fitting_results["w_right"].values, '--o', color='C6', label='s_right')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Weight")
# pl.legend()
pl.subplot(235)
for i in range(4):
    pl.plot(target_df_binned_features_heading_angle_change_histograms.loc[i, :].droplevel("stim"), color=f"black")
    pl.plot(model_df_binned_features_heading_angle_change_histograms.loc[i, :].droplevel("stim"), color=f"C6", linestyle='--')
pl.xlabel("Heading angle change")
pl.ylabel("Probability")
pl.show()


found_parameters = []
for repeat in range(12):
    for genotype in ["wt", "het", "hom"]:
        X = np.load(root_path / f"leaky_integrator_model2_X_{genotype}_{repeat}.npy")
        F = np.load(root_path / f"leaky_integrator_model2_F_{genotype}_{repeat}.npy")

        for i in range(5):
            #F[-1, :, i] = F[-1, :, i] / np.median(F[-1, :, i])
            F[-1, :, i] = F[-1, :, i] / np.percentile(F[-1, :, i], 75)
        #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] + 5 * F[-1, :, 3] + F[-1, :, 5] + 5 * F[-1, :, 6])
        #i6 = np.argmin(F[-1, :, 0] + 5 * F[-1, :, 1] + 20 * F[-1, :, 4] + F[-1, :, 5] + 5 * F[-1, :, 6])
        i6 = np.argmin(F[-1, :, 0] + 3 * F[-1, :, 1] + F[-1, :, 2] + F[-1, :, 3] + F[-1, :, 4])
        #i6 = np.argmin(F[-1, :, 0] + 2 * F[-1, :, 1] + F[-1, :, 2] + 3 * F[-1, :, 3] + F[-1, :, 5] + F[-1, :, 6])
        #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] * 500 + F[-1, :, 3] + F[-1, :, 5] * 0.25 + F[-1, :, 6] * 500)
        #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] * 2500 + F[-1, :, 3] * 5 + F[-1, :, 5] * 0.5 + F[-1, :, 6] * 1500)
        #i6 = np.argmin(F[-1, :, 0]*500 + F[-1, :, 1]*2500 + F[-1, :, 3]*50 + F[-1, :, 5]*0.5 + F[-1, :, 6]*4500)

        found_parameters.append([genotype, repeat, 49] + list(X[-1, i6, :]))

df = pd.DataFrame(found_parameters,
                  columns=["genotype",
                           "repeat",
                           "gen",
                           "tau",
                           "sigma",
                           "T",
                           "p_below",
                           "p_above"]).astype(dtype={"repeat": "int64", "gen": "int64"}, copy=False)

df.set_index(["genotype", 'repeat', 'gen'], inplace=True)
df.sort_index(inplace=True)
df.to_hdf(root_path / "found_parameters.h5", key="parameters", complevel=9)
