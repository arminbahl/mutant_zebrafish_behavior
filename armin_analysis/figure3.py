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
chance_distance = 266.72792263477737
chance_polarization = 0.3169689901297303

fig = myfig.Figure(title=f"Figure 3")

for i, experiment in enumerate(["scn1lab_sa16474", "scn1lab_NIBR", "disc1_hetinx"]):

    polarizations_wt = np.load(root_path / experiment / f"polarizations_wt.npy")[:,::10]
    polarizations_het = np.load(root_path / experiment / f"polarizations_het.npy")[:,::10]

    neighbor_distances_wt = np.load(root_path / experiment / f"neighbor_distances_wt.npy")[:,::10]
    neighbor_distances_het = np.load(root_path / experiment / f"neighbor_distances_het.npy")[:,::10]


    p0 = myfig.Plot(fig, num='a1', xpos=3+ i* 3, ypos=20, plot_height=1.25, plot_width=1.25, title = experiment,
                                   lw=1, pc='white', errorbar_area=True, hlines=[chance_polarization],
                                   xmin=-1, xmax=101, xticks=[0, 20, 40, 60, 80, 100],
                                   yl="Polarisation", ymin=0.2, ymax=1.1, yticks=[0.2, 0.4, 0.6],
                                   vspans=[[20, 80, "lightgray", 0.6]])

    for j in range(12):
        myfig.Line(p0, x=time, y=polarizations_wt[j], lc="black", zorder=1, lw=0.25, alpha=0.25)
        myfig.Line(p0, x=time, y=polarizations_het[j], lc="blue", zorder=1, lw=0.25, alpha=0.25)


    myfig.Line(p0, x=time, y=np.mean(polarizations_wt, axis=0), yerr=sem(polarizations_wt, axis=0), lc="black", zorder=2, label="wt")
    myfig.Line(p0, x=time, y=np.mean(polarizations_het, axis=0), yerr=sem(polarizations_het, axis=0), lc="blue", zorder=2, label="het")


    p0 = myfig.Plot(fig, num='a1', xpos=3+ i* 3, ypos=15, plot_height=1.25, plot_width=1.25, title = "",
                                   lw=1, pc='white', errorbar_area=True, hlines=[chance_distance],
                                   xmin=-1, xmax=101, xticks=[0, 20, 40, 60, 80, 100],
                                   yl="Average neighbor distance", ymin=200, ymax=400, yticks=[200, 300, 400],
                                   vspans=[[20, 80, "lightgray", 0.6]])

    for j in range(12):
        myfig.Line(p0, x=time, y=neighbor_distances_wt[j], lc="black", zorder=1, lw=0.25, alpha=0.25)
        myfig.Line(p0, x=time, y=neighbor_distances_het[j], lc="blue", zorder=1, lw=0.25, alpha=0.25)


    myfig.Line(p0, x=time, y=np.mean(neighbor_distances_wt, axis=0), yerr=sem(neighbor_distances_wt, axis=0), lc="black", zorder=2, label="wt")
    myfig.Line(p0, x=time, y=np.mean(neighbor_distances_het, axis=0), yerr=sem(neighbor_distances_het, axis=0), lc="blue", zorder=2, label="het")


fig.savepdf(target_path / f"raw_figure3", open_pdf=True)
