from pathlib import Path
import numpy as np
from fit_integrator_model import get_model_result, get_target_result
import my_figure as myfig
import matplotlib
import pandas as pd
import pylab as pl
from scipy.stats import sem

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


root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence")
target_path = Path("/Users/arminbahl/Dropbox/pandas_data_for_ariel/paper/figures")

dt = 0.01
time = np.arange(0, 10000*dt, dt)[::10]
time_lowrest = np.arange(0, 99, 1)

chance_distance = 266.72792263477737
chance_polarization = 0.3169689901297303

fig = myfig.Figure(title=f"Figure 3")
colors = ["C3", "red", "C4"]

for i, experiment in enumerate(["scn1lab_NIBR_20200708", "scn1lab_zirc_20200710", "disc1_hetinx"]):

    polarizations_wt = np.load(root_path / experiment / f"polarizations_wt.npy")[:,::10]
    neighbor_distances_wt = np.load(root_path / experiment / f"neighbor_distances_wt.npy")[:,::10]
    speed_over_time_wt = np.load(root_path / experiment / f"speed_over_time_wt.npy")[:, :]

    if experiment == "scn1lab_NIBR_20200708" or experiment == "scn1lab_zirc_20200710":
        polarizations_mutant = np.load(root_path / experiment / f"polarizations_het.npy")[:,::10]
        neighbor_distances_mutant = np.load(root_path / experiment / f"neighbor_distances_het.npy")[:, ::10]
        speed_over_time_mutant = np.load(root_path / experiment / f"speed_over_time_het.npy")[:, :]

    if experiment == "disc1_hetinx":
        polarizations_mutant = np.load(root_path / experiment / f"polarizations_hom.npy")[:,::10]
        neighbor_distances_mutant = np.load(root_path / experiment / f"neighbor_distances_hom.npy")[:, ::10]
        speed_over_time_mutant = np.load(root_path / experiment / f"speed_over_time_hom.npy")[:, :]

    # Speed
    p0 = myfig.Plot(fig, num='a' if i == 0 else '', xpos=3 + i * 2, ypos=15, plot_height=1.25, plot_width=1.5, title = experiment,
                                   lw=1, pc='white', errorbar_area=True, hlines=[chance_distance],
                                   xmin=-1, xmax=101, xticks=[0, 20, 40, 60, 80, 100], xticklabels = [""]*6,
                                   yl="Speed" if i == 0 else None, ymin=-1, ymax=51, yticks=[0, 25, 50] if i == 0 else None,
                                   vspans=[[20, 80, "lightgray", 0.6]])

    for j in range(12):
        myfig.Line(p0, x=time_lowrest, y=speed_over_time_wt[j], lc="black", zorder=1, lw=0.25, alpha=0.25)
        myfig.Line(p0, x=time_lowrest, y=speed_over_time_mutant[j], lc=colors[i], zorder=1, lw=0.25, alpha=0.25)


    myfig.Line(p0, x=time_lowrest, y=np.mean(speed_over_time_wt, axis=0), yerr=sem(speed_over_time_wt, axis=0), lc="black", zorder=2, label="Wildtype" if i == 2 else None)
    myfig.Line(p0, x=time_lowrest, y=np.mean(speed_over_time_mutant, axis=0), yerr=sem(speed_over_time_mutant, axis=0), lc=colors[i], zorder=2, label="Mutant" if i == 2 else None)

    # Group spacing
    p0 = myfig.Plot(fig, num='b' if i == 0 else '', xpos=3 + i * 2, ypos=13, plot_height=1.25, plot_width=1.5, title = "",
                                   lw=1, pc='white', errorbar_area=True, hlines=[chance_distance],
                                   xmin=-1, xmax=101, xticks=[0, 20, 40, 60, 80, 100], xticklabels = [""]*6,
                                   yl="Group spacing" if i == 0 else '', ymin=190, ymax=410, yticks=[200, 300, 400] if i == 0 else None,
                                   vspans=[[20, 80, "lightgray", 0.6]])

    for j in range(12):
        myfig.Line(p0, x=time, y=neighbor_distances_wt[j], lc="black", zorder=1, lw=0.25, alpha=0.25)
        myfig.Line(p0, x=time, y=neighbor_distances_mutant[j], lc=colors[i], zorder=1, lw=0.25, alpha=0.25)


    myfig.Line(p0, x=time, y=np.mean(neighbor_distances_wt, axis=0), yerr=sem(neighbor_distances_wt, axis=0), lc="black", zorder=2)
    myfig.Line(p0, x=time, y=np.mean(neighbor_distances_mutant, axis=0), yerr=sem(neighbor_distances_mutant, axis=0), lc=colors[i], zorder=2)

    # Polarisation
    p0 = myfig.Plot(fig, num='c' if i == 0 else '', xpos=3 + i * 2, ypos=11, plot_height=1.25, plot_width=1.5,
                                   lw=1, pc='white', errorbar_area=True, hlines=[chance_polarization],
                                   xl = "Time (s)", xmin=-1, xmax=101, xticks=[0, 20, 40, 60, 80, 100],
                                   yl="Polarization" if i == 0 else None, ymin=0.15, ymax=0.7, yticks=[0.2, 0.4, 0.6] if i == 0 else None,
                                   vspans=[[20, 80, "lightgray", 0.6]])

    for j in range(12):
        myfig.Line(p0, x=time, y=polarizations_wt[j], lc="black", zorder=1, lw=0.25, alpha=0.25)
        myfig.Line(p0, x=time, y=polarizations_mutant[j], lc=colors[i], zorder=1, lw=0.25, alpha=0.25)


    myfig.Line(p0, x=time, y=np.mean(polarizations_wt, axis=0), yerr=sem(polarizations_wt, axis=0), lc="black", zorder=2)
    myfig.Line(p0, x=time, y=np.mean(polarizations_mutant, axis=0), yerr=sem(polarizations_mutant, axis=0), lc=colors[i], zorder=2)

    ### Speed stats
    p0 = myfig.Plot(fig, num='', xpos=12.0 + i * 1, ypos=15, plot_height=0.75, plot_width=0.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="", xmin=-0.5, xmax=1.5, xticks=[0, 1],
                    yl="Speed" if i == 0 else None, ymin=-1, ymax=51, yticks=[0, 25, 50] if i == 0 else None)

    myfig.Scatter(p0, x=[0]*12, y=np.nanmean(speed_over_time_wt[:, int(20):int(80)], axis=1),
                  lc='black', pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)


    myfig.Scatter(p0, x=[0], y=np.median(np.nanmean(speed_over_time_wt[:, int(20):int(80)], axis=1)),
                  lc='black', pt='o',
                  lw=1, ps=10, pc='white', zorder=2)

    myfig.Scatter(p0, x=[1] * 12, y=np.nanmean(speed_over_time_mutant[:, int(20):int(80)], axis=1),
                  lc=colors[i], pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)

    myfig.Scatter(p0, x=[1], y=np.median(np.nanmean(speed_over_time_mutant[:, int(20):int(80)], axis=1)),
                  lc=colors[i], pt='o',
                  lw=1, ps=10, pc='white', zorder=2)

    p, stars = bootstrap(np.nanmean(speed_over_time_wt[:, int(20):int(80)], axis=1), np.nanmean(speed_over_time_mutant[:, int(20):int(80)], axis=1))
    myfig.Line(p0, x=[0, 1], y=[45, 45], lc='black', lw=0.75)
    myfig.Text(p0, x=0.5, y=48, text=stars)

    ### Spacing stats
    p0 = myfig.Plot(fig, num='', xpos=12.0 + i * 1, ypos=13, plot_height=0.75, plot_width=0.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="", xmin=-0.5, xmax=1.5, xticks=[0, 1],
                    yl="Group spacing" if i == 0 else None, ymin=190, ymax=410, yticks=[200, 300, 400] if i == 0 else None)

    myfig.Scatter(p0, x=[0]*12, y=np.nanmean(neighbor_distances_wt[:, int(20/0.1):int(80/0.1)], axis=1),
                  lc='black', pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)

    myfig.Scatter(p0, x=[0], y=np.median(np.nanmean(neighbor_distances_wt[:, int(20/0.1):int(80/0.1)], axis=1)),
                  lc='black', pt='o',
                  lw=1, ps=10, pc='white', zorder=2)

    myfig.Scatter(p0, x=[1] * 12, y=np.nanmean(neighbor_distances_mutant[:, int(20/0.1):int(80/0.1)], axis=1),
                  lc=colors[i], pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)

    myfig.Scatter(p0, x=[1], y=np.median(np.nanmean(neighbor_distances_mutant[:, int(20/0.1):int(80/0.1)], axis=1)),
                  lc=colors[i], pt='o',
                  lw=1, ps=10, pc='white', zorder=2)

    p, stars = bootstrap(np.nanmean(neighbor_distances_wt[:, int(20/0.1):int(80/0.1)], axis=1), np.nanmean(neighbor_distances_mutant[:, int(20/0.1):int(80/0.1)], axis=1))
    myfig.Line(p0, x=[0, 1], y=[350, 350], lc='black', lw=0.75)
    myfig.Text(p0, x=0.5, y=380, text=stars)

    ### Polaeiation stats
    p0 = myfig.Plot(fig, num='', xpos=12.0 + i * 1, ypos=11, plot_height=0.75, plot_width=0.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="", xmin=-0.5, xmax=1.5, xticks=[0, 1],
                    yl="Polarization" if i == 0 else None, ymin=0.15, ymax=0.7, yticks=[0.2, 0.4, 0.6] if i == 0 else None)

    myfig.Scatter(p0, x=[0]*12, y=np.nanmean(polarizations_wt[:, int(20/0.1):int(80/0.1)], axis=1),
                  lc='black', pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)

    myfig.Scatter(p0, x=[0], y=np.median(np.nanmean(polarizations_wt[:, int(20/0.1):int(80/0.1)], axis=1)),
                  lc='black', pt='o',
                  lw=1, ps=10, pc='white', zorder=2)

    myfig.Scatter(p0, x=[1] * 12, y=np.nanmean(polarizations_mutant[:, int(20/0.1):int(80/0.1)], axis=1),
                  lc=colors[i], pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)

    myfig.Scatter(p0, x=[1], y=np.median(np.nanmean(polarizations_mutant[:, int(20/0.1):int(80/0.1)], axis=1)),
                  lc=colors[i], pt='o',
                  lw=1, ps=10, pc='white', zorder=2)

    p, stars = bootstrap(np.nanmean(polarizations_wt[:, int(20/0.1):int(80/0.1)], axis=1),
                         np.nanmean(polarizations_mutant[:, int(20/0.1):int(80/0.1)], axis=1))
    myfig.Line(p0, x=[0, 1], y=[0.6, 0.6], lc='black', lw=0.75)
    myfig.Text(p0, x=0.5, y=0.7, text=stars)


fig.savepdf(target_path / f"raw_figure3", open_pdf=True)
