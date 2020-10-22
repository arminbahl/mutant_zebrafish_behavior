from pathlib import Path
import numpy as np
from fit_integrator_model import get_model_result, get_target_result
import my_figure as myfig
import matplotlib
import pandas as pd
import pylab as pl
from scipy.stats import sem
from scipy import stats

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

def ttest_1_sided(vals, estimate):
    t, p = stats.ttest_1samp(vals, estimate)
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


#root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/free_swimming_WT_by_age")
root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/model_results")

target_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/paper/figures")

genotype = "hom"

colors = ["C3", "red", "C4"]

fig = myfig.Figure(title=f"Figure 1")

### Speed stats
p0 = myfig.Plot(fig, num='', xpos=12.0, ypos=15, plot_height=0.75, plot_width=0.5,
                lw=1, pc='white', errorbar_area=False, hlines=[0],
                xl="", xmin=-0.5, xmax=2.5, xticks=[0, 1, 2],
                    yl="Speed (cm/s)", ymin=-0.1, ymax=1.1, yticks=[0, 0.5, 1.0] )

### Spacing stats
p1 = myfig.Plot(fig, num='', xpos=12.0, ypos=13, plot_height=0.75, plot_width=0.5,
                lw=1, pc='white', errorbar_area=False, hlines=[0],
                xl="", xmin=-0.5, xmax=2.5, xticks=[0, 1, 2],
                yl="Group spacing\nrel. to chance", ymin=-0.5, ymax=0.5,
                yticks=[-0.5, 0, 0.5])

### Polaeiation stats
p2 = myfig.Plot(fig, num='', xpos=12.0, ypos=11, plot_height=0.75, plot_width=0.5,
                lw=1, pc='white', errorbar_area=False, hlines=[0],
                xl="", xmin=-0.5, xmax=2.5, xticks=[0, 1, 2],
                yl="Polarization\nrel. to chance", ymin=-0.5, ymax=0.5,
                yticks=[-0.5, 0, 0.5])

for i, age in enumerate([7, 14, 21]):

    group_polarization_rel_chance = np.load(root_path / f"group_polarization_rel_chance_{genotype}_age{age}dpf.npy")
    group_spacing_rel_chance = np.load(root_path /  f"group_spacing_rel_chance_{genotype}_age{age}dpf.npy")#[:,::10] / 100
    group_speed = np.load(root_path /  f"group_speed_{genotype}_age{age}dpf.npy")#[:, :] / 100

    myfig.Scatter(p0, x=[i]*group_speed.shape[0], y=group_speed,
                  lc='black', pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)

    myfig.Scatter(p0, x=[i], y=np.median(group_speed),
                  lc='black', pt='o',
                  lw=1, ps=10, pc='white', zorder=2)


    myfig.Scatter(p1, x=[i]*group_spacing_rel_chance.shape[0], y=group_spacing_rel_chance,
                  lc='black', pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)

    myfig.Scatter(p1, x=[i], y=np.median(group_spacing_rel_chance),
                  lc='black', pt='o',
                  lw=1, ps=10, pc='white', zorder=2)


    myfig.Scatter(p2, x=[i]*group_polarization_rel_chance.shape[0], y=group_polarization_rel_chance,
                  lc='black', pt='o',
                  lw=0.5, ps=5, pc='white', zorder=2, alpha=0.5)

    myfig.Scatter(p2, x=[i], y=np.median(group_polarization_rel_chance),
                  lc='black', pt='o',
                  lw=1, ps=10, pc='white', zorder=2)

fig.savepdf(target_path / f"raw_figure1", open_pdf=True)
