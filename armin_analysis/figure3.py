from pathlib import Path
import numpy as np
from fit_integrator_model import get_model_result, get_target_result
import my_figure as myfig
import matplotlib
import pandas as pd
import pylab as pl
from scipy.stats import sem

root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence")
target_path = Path("/Users/arminbahl/Dropbox/pandas_data_for_ariel/paper/figures")

dt = 0.01
time = np.arange(0, 10000*dt, dt)[::10]
time_lowrest = np.arange(0, 99, 1)

chance_distance = 266.72792263477737
chance_polarization = 0.3169689901297303

fig = myfig.Figure(title=f"Figure 3")
colors = ["C3", "red", "C4"]

for i, experiment in enumerate(["scn1lab_sa16474", "scn1lab_NIBR", "disc1_hetinx"]):

    polarizations_wt = np.load(root_path / experiment / f"polarizations_wt.npy")[:,::10]
    neighbor_distances_wt = np.load(root_path / experiment / f"neighbor_distances_wt.npy")[:,::10]
    speed_over_time_wt = np.load(root_path / experiment / f"speed_over_time_wt.npy")[:, :]

    if experiment == "scn1lab_sa16474" or experiment == "scn1lab_NIBR":
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
                                   yl="Group spacing" if i == 0 else '', ymin=200, ymax=400, yticks=[200, 300, 400] if i == 0 else None,
                                   vspans=[[20, 80, "lightgray", 0.6]])

    for j in range(12):
        myfig.Line(p0, x=time, y=neighbor_distances_wt[j], lc="black", zorder=1, lw=0.25, alpha=0.25)
        myfig.Line(p0, x=time, y=neighbor_distances_mutant[j], lc=colors[i], zorder=1, lw=0.25, alpha=0.25)


    myfig.Line(p0, x=time, y=np.mean(neighbor_distances_wt, axis=0), yerr=sem(neighbor_distances_wt, axis=0), lc="black", zorder=2)
    myfig.Line(p0, x=time, y=np.mean(neighbor_distances_mutant, axis=0), yerr=sem(neighbor_distances_mutant, axis=0), lc=colors[i], zorder=2)

    # Polarisation
    p0 = myfig.Plot(fig, num='c' if i == 0 else '', xpos=3 + i* 2, ypos=11, plot_height=1.25, plot_width=1.5,
                                   lw=1, pc='white', errorbar_area=True, hlines=[chance_polarization],
                                   xl = "Time (s)", xmin=-1, xmax=101, xticks=[0, 20, 40, 60, 80, 100],
                                   yl="Polarisation" if i == 0 else None, ymin=0.2, ymax=1.1, yticks=[0.2, 0.4, 0.6] if i == 0 else None,
                                   vspans=[[20, 80, "lightgray", 0.6]])

    for j in range(12):
        myfig.Line(p0, x=time, y=polarizations_wt[j], lc="black", zorder=1, lw=0.25, alpha=0.25)
        myfig.Line(p0, x=time, y=polarizations_mutant[j], lc=colors[i], zorder=1, lw=0.25, alpha=0.25)


    myfig.Line(p0, x=time, y=np.mean(polarizations_wt, axis=0), yerr=sem(polarizations_wt, axis=0), lc="black", zorder=2)
    myfig.Line(p0, x=time, y=np.mean(polarizations_mutant, axis=0), yerr=sem(polarizations_mutant, axis=0), lc=colors[i], zorder=2)





fig.savepdf(target_path / f"raw_figure3", open_pdf=True)
