import pylab as pl
import pandas as pd
from pathlib import Path
import numpy as np
import my_figure as myfig
from scipy import stats

root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence/disc1_hetinx")

df_all_bouts = pd.read_hdf(root_path / "all_data.h5", key="all_bouts").query("genotype == 'wt'").droplevel(["genotype"])
df_all_bouts = df_all_bouts.query("stim == 0").droplevel(["stim"])

fishes_IDs = df_all_bouts.index.get_level_values('fish_ID').unique().values


fig = myfig.Figure(title=f"Figure 2")

p0 = myfig.Plot(fig, num='a', xpos=2, ypos=20, plot_height=2, plot_width=2, title = "Original inter-bout-intervals",
                                       lw=1, pc='white', errorbar_area=False,
                                       xl="Inter-bout-interval (s)", xmin=-0.1, xmax=2.1, xticks=[0., 0.5, 1, 1.5, 2],
                                       yl="Next inter-bout-interval (s)", ymin=-0.1, ymax=2.1, yticks=[0., 0.5, 1, 1.5, 2])

p1 = myfig.Plot(fig, num='a', xpos=5, ypos=20, plot_height=2, plot_width=2, title = "Shuffled inter-bout-intervals",
                                       lw=1, pc='white', errorbar_area=False,
                                       xl="Inter-bout-interval (s)", xmin=-0.1, xmax=2.1, xticks=[0., 0.5, 1, 1.5, 2],
                                       yl="", ymin=-0.1, ymax=2.1, yticks=[0., 0.5, 1, 1.5, 2], yticklabels=[""]*5)

p2 = myfig.Plot(fig, num='b', xpos=8, ypos=20, plot_height=1.25, plot_width=1.25, title = "Bout vs. next bout",
                                       lw=1, pc='white', errorbar_area=False,
                                       xl="", xmin=-0.5, xmax=1.5, xticks=[0, 1], xticklabels=["Original data", "Shuffled"], xticklabels_rotation=45,
                                       yl="Correlation coefficient", ymin=-0.1, ymax=0.6, yticks=[0, 0.25, 0.5])

p3 = myfig.Plot(fig, num='b', xpos=10, ypos=20, plot_height=1.25, plot_width=1.25, title = "Bout vs. second-next bout",
                                       lw=1, pc='white', errorbar_area=False,
                                       xl="", xmin=-0.5, xmax=1.5, xticks=[0, 1], xticklabels=["Original data", "Shuffled"], xticklabels_rotation=45,
                                       yl="", ymin=-0.1, ymax=0.6, yticks=[0, 0.25, 0.5])

p4 = myfig.Plot(fig, num='b', xpos=12, ypos=20, plot_height=1.25, plot_width=1.25, title = "Bout vs. third-next bout",
                                       lw=1, pc='white', errorbar_area=False,
                                       xl="", xmin=-0.5, xmax=1.5, xticks=[0, 1], xticklabels=["Original data", "Shuffled"], xticklabels_rotation=45,
                                       yl="", ymin=-0.1, ymax=0.6, yticks=[0, 0.25, 0.5])


cs_next = []
cs_next_shuffled = []

cs_second_next = []
cs_second_next_shuffled = []

cs_third_next = []
cs_third_next_shuffled = []

pl.figure(figsize=(12,4))

for fish_ID in fishes_IDs:

    all_first_bouts = []
    all_next_bouts = []
    all_second_next_bouts = []
    all_third_next_bouts = []

    all_first_bouts_shuffled = []
    all_next_bouts_shuffled = []
    all_second_next_bouts_shuffled = []
    all_third_next_bouts_shuffled = []

    trials = df_all_bouts.index.get_level_values('trial').unique().values

    for trial in trials:
        df = df_all_bouts.query("fish_ID == @fish_ID and trial == @trial")#.droplevel(["fish_ID", "trial"])

        inter_bout_intervals = df["inter_bout_interval"].values
        all_first_bouts.extend(inter_bout_intervals[:-3])
        all_next_bouts.extend(inter_bout_intervals[1:-2])
        all_second_next_bouts.extend(inter_bout_intervals[2:-1])
        all_third_next_bouts.extend(inter_bout_intervals[3:])

        np.random.shuffle(inter_bout_intervals)

        all_first_bouts_shuffled.extend(inter_bout_intervals[:-3])
        all_next_bouts_shuffled.extend(inter_bout_intervals[1:-2])
        all_second_next_bouts_shuffled.extend(inter_bout_intervals[2:-1])
        all_third_next_bouts_shuffled.extend(inter_bout_intervals[3:])

    c_next = np.corrcoef(all_first_bouts, all_next_bouts)[0,1]
    c_next_shuffled = np.corrcoef(all_first_bouts_shuffled, all_next_bouts_shuffled)[0,1]

    c_second_next = np.corrcoef(all_first_bouts, all_second_next_bouts)[0, 1]
    c_second_next_shuffled = np.corrcoef(all_first_bouts_shuffled, all_second_next_bouts_shuffled)[0, 1]

    c_third_next = np.corrcoef(all_first_bouts, all_third_next_bouts)[0, 1]
    c_third_next_shuffled = np.corrcoef(all_first_bouts_shuffled, all_third_next_bouts_shuffled)[0, 1]


    cs_next.append(c_next)
    cs_next_shuffled.append(c_next_shuffled)

    cs_second_next.append(c_second_next)
    cs_second_next_shuffled.append(c_second_next_shuffled)

    cs_third_next.append(c_third_next)
    cs_third_next_shuffled.append(c_third_next_shuffled)

    all_first_bouts = np.array(all_first_bouts)
    all_next_bouts = np.array(all_next_bouts)
    all_first_bouts_shuffled = np.array(all_first_bouts_shuffled)
    all_next_bouts_shuffled = np.array(all_next_bouts_shuffled)
    ind1 = np.where((all_first_bouts < 2) & (all_next_bouts < 2))
    ind2 = np.where((all_first_bouts_shuffled < 2) & (all_next_bouts_shuffled < 2))

    myfig.Scatter(p0, x=all_first_bouts[ind1], y=all_next_bouts[ind1], lc='C3', pt='.', alpha=0.25, ps=1, lw=0.25, rasterized=True)
    myfig.Scatter(p1, x=all_first_bouts_shuffled[ind2], y=all_next_bouts_shuffled[ind2], lc='gray', pt='.', alpha=0.25, ps=1, lw=0.25, rasterized=True)

    myfig.Scatter(p2, x=[0], y=[c_next], lc='C3', pt='o', ps=1.5, zorder=2)
    myfig.Scatter(p2, x=[1], y=[c_next_shuffled], lc='gray', pt='o', ps=1.5, zorder=2)
    myfig.Line(p2, x=[0,1], y=[c_next, c_next_shuffled], lc='black', lw=0.5, zorder=1)

    myfig.Scatter(p3, x=[0], y=[c_second_next], lc='C3', pt='o', ps=1.5, zorder=2)
    myfig.Scatter(p3, x=[1], y=[c_second_next_shuffled], lc='gray', pt='o', ps=1.5, zorder=2)
    myfig.Line(p3, x=[0, 1], y=[c_second_next, c_second_next_shuffled], lc='black', lw=0.5, zorder=1)

    myfig.Scatter(p4, x=[0], y=[c_third_next], lc='C3', pt='o', ps=1.5, zorder=2)
    myfig.Scatter(p4, x=[1], y=[c_third_next_shuffled], lc='gray', pt='o', ps=1.5, zorder=2)
    myfig.Line(p4, x=[0, 1], y=[c_third_next, c_third_next_shuffled], lc='black', lw=0.5, zorder=1)


print(len(cs_next))
sdf
cs_next = np.array(cs_next)
cs_next_shuffled = np.array(cs_next_shuffled)
cs_second_next = np.array(cs_second_next)
cs_second_next_shuffled = np.array(cs_second_next_shuffled)
cs_third_next = np.array(cs_third_next)
cs_third_next_shuffled = np.array(cs_third_next_shuffled)

print(stats.ttest_1samp(cs_next-cs_next_shuffled, 0))
print(stats.ttest_1samp(cs_second_next-cs_second_next_shuffled, 0))
print(stats.ttest_1samp(cs_third_next-cs_third_next_shuffled, 0))

fig.savepdf("/Users/arminbahl/Desktop/guy_interbout_correlation.pdf", open_pdf=True)
