import my_figure as myfig
import matplotlib
import pandas as pd
import pylab as pl
from pathlib import Path
import numpy as np
import my_figure as myfig
from scipy.stats import ttest_ind


#root_path = Path("/Users/arminbahl/Desktop/preprocessed data/4 fish setups free swimming behavior/dot_motion_coherence8_2")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/scn1lab_sa16474")
root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/surrogate_fish1")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/scn1lab_NIBR")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/immp2l_NIBR")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/immp2l_summer")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/disc1_hetinx")

df = pd.read_hdf(root_path / "found_parameters.h5", key="parameters")
df.to_excel(root_path / "found_parameters.xlsx", sheet_name="parameters")

fig = myfig.Figure(title="Found parameters")

fig.add_text(10.5, 22.2 + 1.5, "Optimal parameters", font=fig.font2)

p0 = myfig.Plot(fig, num='', xpos=1.5, ypos=22, plot_height=1.25, plot_width=1.25,
                lw=1, pc='white', errorbar_area=False,
                xl="", xmin=-1, xmax=3, xticks=[0, 1, 2], xticklabels=["WT", "Het", "Hom"], xticklabels_rotation=45,
                yl="tau", ymin=-0.1, ymax=3.1, yticks=[0, 1.5, 3])

p1 = myfig.Plot(fig, num='', xpos=4.0, ypos=22, plot_height=1.25, plot_width=1.25,
                lw=1, pc='white', errorbar_area=False,
                xl="", xmin=-1, xmax=3, xticks=[0, 1, 2], xticklabels=["WT", "Het", "Hom"], xticklabels_rotation=45,
                yl="sigma", ymin=-0.1, ymax=35, yticks=[0, 15, 30])

p2 = myfig.Plot(fig, num='', xpos=6.5, ypos=22, plot_height=1.25, plot_width=1.25,
                lw=1, pc='white', errorbar_area=False,
                xl="", xmin=-1, xmax=3, xticks=[0, 1, 2], xticklabels=["WT", "Het", "Hom"], xticklabels_rotation=45,
                yl="T", ymin=-0.1, ymax=1.1, yticks=[0, 0.5, 1])

p3 = myfig.Plot(fig, num='', xpos=9, ypos=22, plot_height=1.25, plot_width=1.25,
                lw=1, pc='white', errorbar_area=False,
                xl="", xmin=-1, xmax=3, xticks=[0, 1, 2], xticklabels=["WT", "Het", "Hom"], xticklabels_rotation=45,
                yl="p_below", ymin=-0.005, ymax=0.055, yticks=[0, 0.025, 0.05])

p4 = myfig.Plot(fig, num='', xpos=11.5, ypos=22, plot_height=1.25, plot_width=1.25,
                lw=1, pc='white', errorbar_area=False,
                xl="", xmin=-1, xmax=3, xticks=[0, 1, 2], xticklabels=["WT", "Het", "Hom"], xticklabels_rotation=45,
                yl="p_above", ymin=-0.005, ymax=0.055, yticks=[0, 0.025, 0.05])

p5 = myfig.Plot(fig, num='', xpos=15, ypos=22, plot_height=1.25, plot_width=1.25, title="wt",
                lw=1, pc='white', errorbar_area=False, zmin=-1, zmax=1,
                xl="", xmin=0, xmax=5, xticks=[0.5, 1.5, 2.5, 3.5, 4.5], xticklabels=["tau", "sigma", "T", "p_below", "p_above"], xticklabels_rotation=90,
                yl="", ymin=0, ymax=5, yticks=[0.5, 1.5, 2.5, 3.5, 4.5], yticklabels=["tau", "sigma", "T", "p_below", "p_above"][::-1], yticklabels_rotation=0,
                colormap='bwr')

p6 = myfig.Plot(fig, num='', xpos=16.5, ypos=22, plot_height=1.25, plot_width=1.25, title="het",
                lw=1, pc='white', errorbar_area=False, zmin=-1, zmax=1,
                xl="", xmin=0, xmax=5, xticks=[0.5, 1.5, 2.5, 3.5, 4.5], xticklabels=["tau", "sigma", "T", "p_below", "p_above"], xticklabels_rotation=90,
                yl="", ymin=0, ymax=5, yticks=[0.5, 1.5, 2.5, 3.5, 4.5], yticklabels=["", "", "", "", ""],
                colormap='bwr')

p7 = myfig.Plot(fig, num='', xpos=18, ypos=22, plot_height=1.25, plot_width=1.25, title="hom",
                lw=1, pc='white', errorbar_area=False, zmin=-1, zmax=1,
                xl="", xmin=0, xmax=5, xticks=[0.5, 1.5, 2.5, 3.5, 4.5], xticklabels=["tau", "sigma", "T", "p_below", "p_above"], xticklabels_rotation=90,
                yl="", ymin=0, ymax=5, yticks=[0.5, 1.5, 2.5, 3.5, 4.5], yticklabels=["", "", "", "", ""],
                colormap='bwr', show_colormap=True, zticks=[-1, -0.5, 0, 0.5, 1], zl="Correlation")


def bootstrap(vals1, vals2):

    combined = np.r_[vals1, vals2]
    ds = []
    for i in range(10000):
        ds.append(np.random.choice(combined, 12).mean() - np.random.choice(combined, 12).mean())

    ds = np.array(ds)
    d_real = np.abs(vals1.mean() - vals2.mean())

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

    # print(p)
    # pl.figure()
    # pl.hist(ds, bins=50)
    # pl.axvline(d_real, 0, 10)
    # pl.show()
    # sdf

for i, genotype, lc in zip([0, 1, 2], ["wt", "het", "hom"], ["black", "firebrick", "blue"]):

    for repeat in range(12):

        df_selected = df.query("genotype == @genotype and repeat == @repeat and gen == 49")
        if len(df_selected) == 0:
            continue

        myfig.Scatter(p0, x=[i], y=df_selected["tau"], lc=lc, pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)

        myfig.Scatter(p1, x=[i], y=df_selected["sigma"], lc=lc, pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)

        myfig.Scatter(p2, x=[i], y=df_selected["T"], lc=lc, pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)

        myfig.Scatter(p3, x=[i], y=df_selected["p_below"], lc=lc, pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)

        myfig.Scatter(p4, x=[i], y=df_selected["p_above"], lc=lc, pt='o',
                      lw=0.5, ps=9.8, pc='white', zorder=2, alpha=0.5)

    print(df.query("genotype == @genotype and gen == 49").mean(axis=0))
    corr_matrix = df.query("genotype == @genotype and gen == 49").corr()
    myfig.Mat([p5, p6, p7][i], corr_matrix, extent=(0, 5, 0, 5), interpolation='nearest', origin='upper')


for i, parameter in enumerate(["tau", "sigma", "T", "p_below", "p_above"]):

    values_wt = df.query("genotype == 'wt' and gen == 49")[parameter].values
    values_het = df.query("genotype == 'het' and gen == 49")[parameter].values
    values_hom = df.query("genotype == 'hom' and gen == 49")[parameter].values


    p, stars = bootstrap(values_wt, values_het)
    plot = [p0, p1, p2, p3, p4][i]
    y = [2, 20, 1, 0.025, 0.05][i]
    myfig.Line(plot, x=[0.1, 0.9], y=[y, y], lc='black', lw=0.75)
    myfig.Text(plot, 0.5, y*1.05, stars)

    p, stars = bootstrap(values_wt, values_hom)
    plot = [p0, p1, p2, p3, p4][i]
    y = [2, 20, 1, 0.025, 0.05][i]
    myfig.Line(plot, x=[1.1, 1.9], y=[y, y], lc='black', lw=0.75)
    myfig.Text(plot, 1.5, y*1.05, stars)


fig.savepdf(root_path / f"found_parameters", open_pdf=True)