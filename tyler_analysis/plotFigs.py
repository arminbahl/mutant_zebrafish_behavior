import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.container as container
import json
from json import JSONEncoder

# # names=[u'fish_ID', u'genotype', u'trial', u'stim'],
# # Index[u'bout_time', u'bout_x', u'bout_y', u'inter_bout_interval', u'heading_angle_change', u'same_as_previous'


#############################################################################
################################### PLOTS ###################################
#############################################################################
def p1(gen, fileIndex, printPDF):
    #################### SET UP ####################
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(4,7.25))
    if fileIndex == 0: # Access disc1_hetinx data
        fig.suptitle('\ndisc1_hetinx ' + gen + ' Fig1')
    elif fileIndex == 1: # Access immp2l_NIBR data
        fig.suptitle('\nimmp2l_NIBR ' + gen + ' Fig1')
    elif fileIndex == 2: # Access immp2l_summer data
        fig.suptitle('\nimmp2l_summer ' + gen + ' Fig1')
    elif fileIndex == 3: # Access scn1lab_NIBR data
        fig.suptitle('\nscn1lab_NIBR ' + gen + ' Fig1')
    elif fileIndex == 4: # Access scn1lab_sa16474 data
        fig.suptitle('\nscn1lab_sa16474 ' + gen + ' Fig1')
    elif fileIndex == 5: # Access surrogate_fish1 data
        fig.suptitle('\nsurrogate_fish1 ' + gen + ' Fig1')
    elif fileIndex == 6: # Access surrogate_fish2 data
        fig.suptitle('\nsurrogate_fish2 ' + gen + ' Fig1')
    elif fileIndex == 7: # Access surrogate_fish3 data
        fig.suptitle('\nsurrogate_fish3 ' + gen + ' Fig1')
    elif fileIndex == 8: # Access scn1lab_NIBR_20200708 data
        fig.suptitle('\nscn1lab_NIBR_20200708 ' + gen + ' Fig1')
    elif fileIndex == 9: # Access scn1lab_zirc_20200710 data
        fig.suptitle('\nscn1lab_zirc_20200710 ' + gen + ' Fig1')

    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0, hspace=0.12)

    coherence = [0, 25, 50, 100]
    if gen == "het":
        gen_color = "purple"
    elif gen == "hom":
        gen_color = "darkblue"

    data = loadData(1)

    #################### PLOT 1A ####################
    # 1A: Swims per second as a function of coherence (wt and gen)
    wt1 = np.asarray(getData("oneA", "wt", fileIndex, data)[1])
    gen1 = np.asarray(getData("oneA", gen, fileIndex, data)[1])

    ax1.set_xticks(coherence)
    ax1.set_ylim([0, 2])
    ax1.set(ylabel="swims per second")

    ax1.errorbar(coherence, wt1[0], marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                  label="wt", yerr=wt1[1], capsize=4, capthick=1)
    ax1.errorbar(coherence, gen1[0], marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color,
                  ls="--", label=gen, yerr=gen1[1], capsize=4, capthick=1)

    handles, labels = ax1.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax1.legend(handles, labels, loc="upper left", frameon=False, prop={'size': 8})

    #################### PLOT 1B ####################
    # 1B: Fraction of correct bouts as a function of coherence (wt and gen)
    wt2 = np.asarray(getData("oneB", "wt", fileIndex, data)[1])
    gen2 = np.asarray(getData("oneB", gen, fileIndex, data)[1])

    ax2.set_xticks(coherence)
    ax2.set_ylim([20, 100])
    ax2.set(ylabel="fraction of correct bouts (%)")

    ax2.errorbar(coherence, wt2[0], marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                 label="wt", yerr=wt2[1], capsize=4, capthick=1)
    ax2.errorbar(coherence, gen2[0], marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color,
                 ls="--", label=gen, yerr=gen2[1], capsize=4, capthick=1)

    #################### PLOT 1C ####################
    # 1C: Angle turned per correct bout as a function of coherence (wt and gen)
    wt3 = np.asarray(getData("oneC", "wt", fileIndex, data)[1])
    gen3 = np.asarray(getData("oneC", gen, fileIndex, data)[1])

    ax3.set_xticks(coherence)
    ax3.set_ylim([0, 50])
    ax3.set(ylabel="angle turned \nper correct bout (\N{DEGREE SIGN})", xlabel="% coherence")

    ax3.errorbar(coherence, wt3[0], marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                 label="wt", yerr=wt3[1], capsize=4, capthick=1)
    ax3.errorbar(coherence, gen3[0], marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color,
                 ls="--", label=gen, yerr=gen3[1], capsize=4, capthick=1)

    #################### FINISH ####################
    if printPDF:
        figPDF = plt.gcf()
        plt.draw()

        if fileIndex == 0: # Access disc1_hetinx data
            figPDF.savefig('figs1/disc1_hetinx_'+gen+'_Fig1.pdf')
        elif fileIndex == 1: # Access immp2l_NIBR data
            figPDF.savefig('figs1/immp2l_NIBR_'+gen+'_Fig1.pdf')
        elif fileIndex == 2: # Access immp2l_summer data
            figPDF.savefig('figs1/immp2l_summer_'+gen+'_Fig1.pdf')
        elif fileIndex == 3: # Access scn1lab_NIBR data
            figPDF.savefig('figs1/scn1lab_NIBR_'+gen+'_Fig1.pdf')
        elif fileIndex == 4: # Access scn1lab_sa16474 data
            figPDF.savefig('figs1/scn1lab_sa16474_'+gen+'_Fig1.pdf')
        elif fileIndex == 5: # Access surrogate_fish1 data
            figPDF.savefig('figs1/surrogate_fish1_'+gen+'_Fig1.pdf')
        elif fileIndex == 6: # Access surrogate_fish2 data
            figPDF.savefig('figs1/surrogate_fish2_'+gen+'_Fig1.pdf')
        elif fileIndex == 7: # Access surrogate_fish3 data
            figPDF.savefig('figs1/surrogate_fish3_'+gen+'_Fig1.pdf')
        elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
            figPDF.savefig('figs1/scn1lab_NIBR_20200708_'+gen+'_Fig1.pdf')
        elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
            figPDF.savefig('figs1/scn1lab_zirc_20200710_'+gen+'_Fig1.pdf')
    else:
        plt.show()

def p2(gen, fileIndex, printPDF):
    #################### SET UP ####################
    fig = plt.figure(figsize=(12, 3.5))
    fig.subplots_adjust(bottom=0.15)
    outer = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.2, height_ratios=(0.1,9.9))
    inner1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[4], wspace=0.1, hspace=0.1)
    inner2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[5], wspace=0.1, hspace=0.1)
    inner3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[6], wspace=0.1, hspace=0.1, width_ratios=(6.5,4))
    inner4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[7], wspace=0.1, hspace=0.1)
    ax1a = plt.Subplot(fig, inner1[0])
    ax1b = plt.Subplot(fig, inner1[1])
    ax2a = plt.Subplot(fig,inner2[0])
    ax2b = plt.Subplot(fig, inner2[1])
    ax3a = plt.Subplot(fig,inner3[0])
    ax3b = plt.Subplot(fig, inner3[1])
    ax3c = plt.Subplot(fig, inner3[2])
    ax3d = plt.Subplot(fig, inner3[3])
    ax4 = plt.Subplot(fig, inner4[0])
    subplot_list = [ax1a,ax1b,ax2a,ax2b,ax3a,ax3b,ax3c,ax3d,ax4]
    for subplt in subplot_list:
        fig.add_subplot(subplt)

    if fileIndex == 0:  # Access disc1_hetinx data
        fig.suptitle('\ndisc1_hetinx ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 1:  # Access immp2l_NIBR data
        fig.suptitle('\nimmp2l_NIBR ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 2:  # Access immp2l_summer data
        fig.suptitle('\nimmp2l_summer ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 3:  # Access scn1lab_NIBR data
        fig.suptitle('\nscn1lab_NIBR ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 4:  # Access scn1lab_sa16474 data
        fig.suptitle('\nscn1lab_sa16474 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 5:  # Access surrogate_fish1 data
        fig.suptitle('\nsurrogate_fish1 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 6:  # Access surrogate_fish2 data
        fig.suptitle('\nsurrogate_fish2 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 7:  # Access surrogate_fish3 data
        fig.suptitle('\nsurrogate_fish3 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 8: # Access scn1lab_NIBR_20200708 data
        fig.suptitle('\nscn1lab_NIBR_20200708 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 9: # Access scn1lab_zirc_20200710 data
        fig.suptitle('\nscn1lab_zirc_20200710 ' + gen + ' Fig2', y=0.96)

    coherence = [0, 25, 50, 100]
    if gen == "het":
        gen_color = "purple"
    elif gen == "hom":
        gen_color = "darkblue"

    data = loadData(2)

    #################### PLOT 2Aa ####################
    # 2Aa: Probability correct as a function of coherence (wt and gen)
    wt1a = np.asarray(getData("twoAa", "wt", fileIndex, data)[1])
    gen1a = np.asarray(getData("twoAa", gen, fileIndex, data)[1])
    wt1a_val = wt1a[0]
    gen1a_val = gen1a[0]
    wt1a_sem = wt1a[1]
    gen1a_sem = gen1a[1]

    ax1a.set_ylim([40, 100])
    ax1a.set_yticks([50, 70, 90])
    ax1a.set_xticks(coherence)
    ax1a.tick_params(labelbottom=False)
    ax1a.set(ylabel="Probability\ncorrect (%)")
    ax1a.spines["top"].set_visible(False)
    ax1a.spines["right"].set_visible(False)

    ax1a.errorbar(coherence, wt1a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black", label="wt", yerr=wt1a_sem, capsize=4, capthick=1)
    ax1a.errorbar(coherence, gen1a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color, label=gen, yerr=gen1a_sem, capsize=4, capthick=1)
    ax1a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    handles, labels = ax1a.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax1a.legend(handles, labels, loc="upper left", frameon=False, prop={'size': 8})

    #################### PLOT 2Ab ####################
    # 2Ab: Interbout interval as a function of coherence (wt and gen)
    wt1b = np.asarray(getData("twoAb", "wt", fileIndex, data)[1])
    gen1b = np.asarray(getData("twoAb", gen, fileIndex, data)[1])
    wt1b_val = wt1b[0]
    gen1b_val = gen1b[0]
    wt1b_sem = wt1b[1]
    gen1b_sem = gen1b[1]

    ax1b.set_ylim([0.25, 2.25])
    ax1b.set_yticks([0.5,1.0,1.5,2.0])
    ax1b.set_xticks(coherence)
    ax1b.set(ylabel="Inter bout\ninterval (s)",xlabel="Coherence (%)")
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["right"].set_visible(False)

    ax1b.errorbar(coherence, wt1b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black", label="wt", yerr=wt1b_sem, capsize=4, capthick=1)
    ax1b.errorbar(coherence, gen1b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color, label=gen, yerr=gen1b_sem, capsize=4, capthick=1)

    #################### PLOT 2B ####################
    # 2B: Time-binned probability correct (wt or gen)
    wt2 = np.asarray(getData("twoB", "wt", fileIndex, data)[1])
    gen2 = np.asarray(getData("twoB", gen, fileIndex, data)[1])
    wt2_val = wt2[0]
    gen2_val = gen2[0]
    wt2_sem = wt2[1]
    gen2_sem = gen2[1]

    ax2a.text(3, 38, "Probability correct (%)", rotation="vertical", va="center")
    ax2_colors = ["black", "maroon", "firebrick", "indianred"]

    ax2a.set_ylim([40, 100])
    ax2a.set_yticks([50, 70, 90])
    ax2b.set_xlim([7, 25])
    ax2a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax2a.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8,0.8,0.8)))
    ax2a.spines["top"].set_visible(False)
    ax2a.spines["bottom"].set_visible(False)
    ax2a.spines["right"].set_visible(False)

    ax2b.set_ylim([40, 100])
    ax2b.set_yticks([50, 70, 90])
    ax2b.set_xlim([7, 25])
    ax2b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax2b.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["bottom"].set_visible(False)
    ax2b.spines["right"].set_visible(False)
    ax2b.text(11.25, 18, "(bin = 2 s)", )

    for i in range(0,4):
        ax2a.errorbar(np.linspace(8, 24, num=len(wt2_val[i])), wt2_val[i], marker="o", markersize=5, markerfacecolor="white",
                     linewidth=1.5, color=ax2_colors[i], label="wt", yerr=wt2_sem[i], capsize=4, capthick=1)

    ax2a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2a.text(23, 94, "wt", fontsize=8)

    for i in range(0,4):
        ax2b.errorbar(np.linspace(8, 24, num=len(gen2_val[i])), gen2_val[i], marker="o", markersize=5,
                      markerfacecolor="white", linewidth=1.5, color=ax2_colors[i], label=gen, yerr=gen2_sem[i], capsize=4, capthick=1)

    ax2b.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2b.text(23, 94, gen, fontsize=8)

    ax2b.plot(np.linspace(20.25,25),np.linspace(20.25,25)*0+40.5, color="black")
    ax2b.text(21.8, 33, "5 s", fontsize=8)
    bracket(ax2b, text="Stimulus period", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),linekw=dict(color="black", lw=1.5))

    #################### PLOT 2C ####################
    # 2C: Probability correct of consecutive bouts after stimulus start and end (wt or gen)
    wt3 = np.asarray(getData("twoC", "wt", fileIndex, data)[1])
    gen3 = np.asarray(getData("twoC", gen, fileIndex, data)[1])
    wt3_val = wt3[0]
    gen3_val = gen3[0]
    wt3_sem = wt3[1]
    gen3_sem = gen3[1]

    ax3a.text(-1.5, 38, "Probability correct (%)", rotation="vertical", va="center")
    ax3_colors = ["black", "maroon", "firebrick", "indianred"]

    ax3a.set_ylim([40, 100])
    ax3a.set_yticks([50, 70, 90])
    ax3a.set_xlim([0.5, 5.5])
    ax3a.set_xticks([1, 2, 3, 4, 5])
    ax3a.tick_params(labelbottom=False)
    ax3a.spines["top"].set_visible(False)
    ax3a.spines["right"].set_visible(False)
    ax3a.add_patch(matplotlib.patches.Rectangle((0.5, 5.5), 5.5, 100, color=(0.8,0.8,0.8)))
    ax3a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    ax3b.set_ylim([40, 100])
    ax3b.set_xlim([0.5, 3.5])
    ax3b.set_xticks([1, 2, 3])
    ax3b.tick_params(labelbottom=False)
    ax3b.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax3b.spines["top"].set_visible(False)
    ax3b.spines["right"].set_visible(False)
    ax3b.spines["left"].set_visible(False)
    ax3b.add_patch(matplotlib.patches.Rectangle((0.3, 1), 0.5, 100, color=(0.8,0.8,0.8)))
    ax3b.text(2.7, 94, "wt", fontsize=8)
    ax3b.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    ax3c.set_ylim([40, 100])
    ax3c.set_yticks([50, 70, 90])
    ax3c.set_xlim([0.5, 5.5])
    ax3c.set_xticks([1, 2, 3, 4, 5])
    ax3c.spines["top"].set_visible(False)
    ax3c.spines["right"].set_visible(False)
    ax3c.add_patch(matplotlib.patches.Rectangle((0.5, 5.5), 5.5, 100, color=(0.8,0.8,0.8)))
    ax3c.set_xlabel("Bout no. since\nstimulus start", fontsize=8)
    ax3c.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    ax3d.set_ylim([40, 100])
    ax3d.set_xlim([0.5, 3.5])
    ax3d.set_xticks([1, 2, 3])
    ax3d.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax3d.spines["top"].set_visible(False)
    ax3d.spines["right"].set_visible(False)
    ax3d.spines["left"].set_visible(False)
    ax3d.add_patch(matplotlib.patches.Rectangle((0.3, 1), 0.5, 100, color=(0.8,0.8,0.8)))
    ax3d.text(2.7, 94, gen, fontsize=8)
    ax3d.set_xlabel("Bout no. since\nstimulus end", fontsize=8)
    ax3d.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    for i in range(0,4):
        ax3a.errorbar([1, 2, 3, 4, 5], wt3_val[i][0:5], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax3_colors[i], label="wt", yerr=wt3_sem[i][0:5], capsize=4, capthick=1)
        ax3b.errorbar([1, 2, 3], wt3_val[i][5:8], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax3_colors[i], label="wt", yerr=wt3_sem[i][5:8], capsize=4, capthick=1)
        ax3c.errorbar([1, 2, 3, 4, 5], gen3_val[i][0:5], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax3_colors[i], label=gen, yerr=gen3_sem[i][0:5], capsize=4, capthick=1)
        ax3d.errorbar([1, 2, 3], gen3_val[i][5:8], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax3_colors[i], label=gen, yerr=gen3_sem[i][5:8], capsize=4, capthick=1)

    #################### PLOT 2D ####################
    # 2D: Probability to swim in same direction as a function of interbout interval (0% coherence, wt and gen)
    wt4 = np.asarray(getData("twoD", "wt", fileIndex, data)[1])
    gen4 = np.asarray(getData("twoD", gen, fileIndex, data)[1])
    wt4_val = wt4[0]
    gen4_val = gen4[0]
    wt4_sem = wt4[1]
    gen4_sem = gen4[1]

    ax4.set_ylim([40, 100])
    ax4.set_yticks([50, 70, 90])
    ax4.set_xlim([-0.4, 3.1])
    ax4.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_title("0% Coherence")
    ax4.set_ylabel("Probability to swim in same direction (%)", fontsize=9)
    ax4.set_xlabel("Inter bout interval\n(bin = 0.5 s)", fontsize=9)

    ax4.errorbar(np.linspace(0, 3, len(wt4_val)), wt4_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black", label="wt", yerr=wt4_sem, capsize=4, capthick=1)
    ax4.errorbar(np.linspace(0, 3, len(gen4_val)), gen4_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color, label=gen, yerr=gen4_sem, capsize=4, capthick=1)
    ax4.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax4.axvline(x=-0.2, linestyle="dotted", color="black", alpha=0.5)

    handles, labels = ax4.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax4.legend(handles, labels, loc="upper right", frameon=False, prop={'size': 8})

    ax4.annotate('Time of\nlast bout', xy=(-0.2, 40), xycoords='data', xytext=(-0.2, 31), textcoords='data',
                  arrowprops=dict(arrowstyle='-|>', color='black', lw=2), fontsize=8, horizontalalignment="center")

    #################### FINISH ####################
    if printPDF:
        figPDF = plt.gcf()
        plt.draw()

        if fileIndex == 0:  # Access disc1_hetinx data
            figPDF.savefig('figs2/disc1_hetinx_' + gen + '_Fig2.pdf')
        elif fileIndex == 1:  # Access immp2l_NIBR data
            figPDF.savefig('figs2/immp2l_NIBR_' + gen + '_Fig2.pdf')
        elif fileIndex == 2:  # Access immp2l_summer data
            figPDF.savefig('figs2/immp2l_summer_' + gen + '_Fig2.pdf')
        elif fileIndex == 3:  # Access scn1lab_NIBR data
            figPDF.savefig('figs2/scn1lab_NIBR_' + gen + '_Fig2.pdf')
        elif fileIndex == 4:  # Access scn1lab_sa16474 data
            figPDF.savefig('figs2/scn1lab_sa16474_' + gen + '_Fig2.pdf')
        elif fileIndex == 5:  # Access surrogate_fish1 data
            figPDF.savefig('figs2/surrogate_fish1_' + gen + '_Fig2.pdf')
        elif fileIndex == 6:  # Access surrogate_fish2 data
            figPDF.savefig('figs2/surrogate_fish2_' + gen + '_Fig2.pdf')
        elif fileIndex == 7:  # Access surrogate_fish3 data
            figPDF.savefig('figs2/surrogate_fish3_' + gen + '_Fig2.pdf')
        elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
            figPDF.savefig('figs2/scn1lab_NIBR_20200708_'+gen+'_Fig2.pdf')
        elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
            figPDF.savefig('figs2/scn1lab_zirc_20200710_'+gen+'_Fig2.pdf')
    else:
        plt.show()

def p2a(gen, fileIndex, printPDF):
    #################### SET UP ####################
    fig = plt.figure(figsize=(12, 5))
    fig.subplots_adjust(bottom=0.15)
    outer = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.4, height_ratios=(0.2, 4.95, 4.95))
    inner1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[3], wspace=0.1, hspace=0.1)
    inner2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[4], wspace=0.1, hspace=0.1)
    inner3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[5], wspace=0.1, hspace=0.1)
    inner4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[6], wspace=0.1, hspace=0.1,
                                              width_ratios=(6.5, 4))
    inner5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[7], wspace=0.1, hspace=0.1,
                                              width_ratios=(6.5, 4))
    inner6 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[8], wspace=0.1, hspace=0.1,
                                              width_ratios=(6.5, 4))
    ax1a = plt.Subplot(fig, inner1[0])
    ax1b = plt.Subplot(fig, inner2[0])
    ax1c = plt.Subplot(fig, inner3[0])
    ax2a1 = plt.Subplot(fig, inner4[0])
    ax2a2 = plt.Subplot(fig, inner4[1])
    ax2b1 = plt.Subplot(fig, inner5[0])
    ax2b2 = plt.Subplot(fig, inner5[1])
    ax2c1 = plt.Subplot(fig, inner6[0])
    ax2c2 = plt.Subplot(fig, inner6[1])
    subplot_list = [ax1a, ax1b, ax1c, ax2a1, ax2a2, ax2b1, ax2b2, ax2c1, ax2c2]
    for subplt in subplot_list:
        fig.add_subplot(subplt)

    if fileIndex == 0:  # Access disc1_hetinx data
        fig.suptitle('\ndisc1_hetinx ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 1:  # Access immp2l_NIBR data
        fig.suptitle('\nimmp2l_NIBR ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 2:  # Access immp2l_summer data
        fig.suptitle('\nimmp2l_summer ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 3:  # Access scn1lab_NIBR data
        fig.suptitle('\nscn1lab_NIBR ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 4:  # Access scn1lab_sa16474 data
        fig.suptitle('\nscn1lab_sa16474 ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 5:  # Access surrogate_fish1 data
        fig.suptitle('\nsurrogate_fish1 ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 6:  # Access surrogate_fish2 data
        fig.suptitle('\nsurrogate_fish2 ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 7:  # Access surrogate_fish3 data
        fig.suptitle('\nsurrogate_fish3 ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
        fig.suptitle('\nscn1lab_NIBR_20200708 ' + gen + ' Fig2a', y=0.96)
    elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
        fig.suptitle('\nscn1lab_zirc_20200710 ' + gen + ' Fig2a', y=0.96)

    coherence = [0, 25, 50, 100]
    coherence_colors = ["black", "maroon", "firebrick", "indianred"]
    if gen == "het":
        gen_color = "purple"
    elif gen == "hom":
        gen_color = "darkblue"

    data = loadData(2)

    #################### PLOT 5A ####################
    # 5A: Time-binned probability correct (wt or gen)
    wt1 = np.asarray(getData("twoB", "wt", fileIndex, data)[1])
    gen1 = np.asarray(getData("twoB", gen, fileIndex, data)[1])

    wt1_val = wt1[0]
    gen1_val = gen1[0]
    wt1_sem = wt1[1]
    gen1_sem = gen1[1]

    ax1a.set_ylim([40, 100])
    ax1a.set_yticks([50, 70, 90])
    ax1a.set_xlim([7, 25])
    ax1a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1a.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax1a.spines["top"].set_visible(False)
    ax1a.spines["bottom"].set_visible(False)
    ax1a.spines["right"].set_visible(False)
    ax1a.set_ylabel("Probability correct (%)")
    ax1a.set_title("25% Coherence")

    ax1b.set_ylim([40, 100])
    ax1b.set_yticks([50, 70, 90])
    ax1b.set_xlim([7, 25])
    ax1b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1b.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["bottom"].set_visible(False)
    ax1b.spines["right"].set_visible(False)
    ax1b.set_title("50% Coherence")

    ax1c.set_ylim([40, 100])
    ax1c.set_yticks([50, 70, 90])
    ax1c.set_xlim([7, 25])
    ax1c.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1c.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax1c.spines["top"].set_visible(False)
    ax1c.spines["bottom"].set_visible(False)
    ax1c.spines["right"].set_visible(False)
    ax1c.set_title("100% Coherence")

    handles, labels = ax1c.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax1c.legend(handles, labels, loc="upper right", frameon=False, prop={'size': 8})

    ax1a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax1b.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax1c.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    ax1a.plot(np.linspace(20.25, 25), np.linspace(20.25, 25) * 0 + 40.5, color="black")
    ax1a.text(22, 34, "5 s", fontsize=8)
    bracket(ax1a, text="Stimulus period", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),
            linekw=dict(color="black", lw=1.5))
    ax1b.plot(np.linspace(20.25, 25), np.linspace(20.25, 25) * 0 + 40.5, color="black")
    ax1b.text(22, 34, "5 s", fontsize=8)
    bracket(ax1b, text="Stimulus period", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),
            linekw=dict(color="black", lw=1.5))
    ax1c.plot(np.linspace(20.25, 25), np.linspace(20.25, 25) * 0 + 40.5, color="black")
    ax1c.text(22, 34, "5 s", fontsize=8)
    bracket(ax1c, text="Stimulus period", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),
            linekw=dict(color="black", lw=1.5))

    ax1a.errorbar(np.linspace(8, 24, num=len(wt1_val[1])), wt1_val[1], marker="o", markersize=5,
                  markerfacecolor="white",
                  linewidth=1.5, color="black", label="wt", yerr=wt1_sem[1], capsize=4, capthick=1)
    ax1a.errorbar(np.linspace(8, 24, num=len(gen1_val[1])), gen1_val[1], marker="o", markersize=5,
                  markerfacecolor="white",
                  linewidth=1.5, color=coherence_colors[1], label=gen, yerr=gen1_sem[1], capsize=4, capthick=1)

    ax1b.errorbar(np.linspace(8, 24, num=len(wt1_val[2])), wt1_val[2], marker="o", markersize=5,
                  markerfacecolor="white",
                  linewidth=1.5, color="black", label="wt", yerr=wt1_sem[2], capsize=4, capthick=1)
    ax1b.errorbar(np.linspace(8, 24, num=len(gen1_val[2])), gen1_val[2], marker="o", markersize=5,
                  markerfacecolor="white",
                  linewidth=1.5, color=coherence_colors[2], label=gen, yerr=gen1_sem[2], capsize=4, capthick=1)

    ax1c.errorbar(np.linspace(8, 24, num=len(wt1_val[3])), wt1_val[3], marker="o", markersize=5,
                  markerfacecolor="white",
                  linewidth=1.5, color="black", label="wt", yerr=wt1_sem[3], capsize=4, capthick=1)
    ax1c.errorbar(np.linspace(8, 24, num=len(gen1_val[3])), gen1_val[3], marker="o", markersize=5,
                  markerfacecolor="white",
                  linewidth=1.5, color=coherence_colors[3], label=gen, yerr=gen1_sem[3], capsize=4, capthick=1)

    #################### PLOT 5B ####################
    # 5B: Probability correct of consecutive bouts after stimulus start and end (wt or gen)
    wt2 = np.asarray(getData("twoC", "wt", fileIndex, data)[1])
    gen2 = np.asarray(getData("twoC", gen, fileIndex, data)[1])

    wt2_val = wt2[0]
    gen2_val = gen2[0]
    wt2_sem = wt2[1]
    gen2_sem = gen2[1]

    ax2a1.set_ylim([40, 100])
    ax2a1.set_yticks([50, 70, 90])
    ax2a1.set_xlim([0.5, 5.5])
    ax2a1.set_xticks([1, 2, 3, 4, 5])
    ax2a1.spines["top"].set_visible(False)
    ax2a1.spines["right"].set_visible(False)
    ax2a1.add_patch(matplotlib.patches.Rectangle((0.5, 5.5), 5.5, 100, color=(0.8, 0.8, 0.8)))
    ax2a1.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2a1.set_ylabel("Probability correct (%)")
    ax2a1.set_xlabel("Bout no. since\nstimulus start", fontsize=8)

    ax2a2.set_ylim([40, 100])
    ax2a2.set_xlim([0.5, 3.5])
    ax2a2.set_xticks([1, 2, 3])
    ax2a2.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2a2.spines["top"].set_visible(False)
    ax2a2.spines["right"].set_visible(False)
    ax2a2.spines["left"].set_visible(False)
    ax2a2.add_patch(matplotlib.patches.Rectangle((0.3, 1), 0.5, 100, color=(0.8, 0.8, 0.8)))
    ax2a2.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2a2.set_xlabel("Bout no. since\nstimulus end", fontsize=8)

    ax2b1.set_ylim([40, 100])
    ax2b1.set_yticks([50, 70, 90])
    ax2b1.set_xlim([0.5, 5.5])
    ax2b1.set_xticks([1, 2, 3, 4, 5])
    ax2b1.spines["top"].set_visible(False)
    ax2b1.spines["right"].set_visible(False)
    ax2b1.add_patch(matplotlib.patches.Rectangle((0.5, 5.5), 5.5, 100, color=(0.8, 0.8, 0.8)))
    ax2b1.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2b1.set_xlabel("Bout no. since\nstimulus start", fontsize=8)

    ax2b2.set_ylim([40, 100])
    ax2b2.set_xlim([0.5, 3.5])
    ax2b2.set_xticks([1, 2, 3])
    ax2b2.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2b2.spines["top"].set_visible(False)
    ax2b2.spines["right"].set_visible(False)
    ax2b2.spines["left"].set_visible(False)
    ax2b2.add_patch(matplotlib.patches.Rectangle((0.3, 1), 0.5, 100, color=(0.8, 0.8, 0.8)))
    ax2b2.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2b2.set_xlabel("Bout no. since\nstimulus end", fontsize=8)

    ax2c1.set_ylim([40, 100])
    ax2c1.set_yticks([50, 70, 90])
    ax2c1.set_xlim([0.5, 5.5])
    ax2c1.set_xticks([1, 2, 3, 4, 5])
    ax2c1.spines["top"].set_visible(False)
    ax2c1.spines["right"].set_visible(False)
    ax2c1.add_patch(matplotlib.patches.Rectangle((0.5, 5.5), 5.5, 100, color=(0.8, 0.8, 0.8)))
    ax2c1.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2c1.set_xlabel("Bout no. since\nstimulus start", fontsize=8)

    ax2c2.set_ylim([40, 100])
    ax2c2.set_xlim([0.5, 3.5])
    ax2c2.set_xticks([1, 2, 3])
    ax2c2.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2c2.spines["top"].set_visible(False)
    ax2c2.spines["right"].set_visible(False)
    ax2c2.spines["left"].set_visible(False)
    ax2c2.add_patch(matplotlib.patches.Rectangle((0.3, 1), 0.5, 100, color=(0.8, 0.8, 0.8)))
    ax2c2.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2c2.set_xlabel("Bout no. since\nstimulus end", fontsize=8)

    ax2a1.errorbar([1, 2, 3, 4, 5], wt2_val[1][0:5], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color="black", label="wt", yerr=wt2_sem[1][0:5], capsize=4, capthick=1)
    ax2a1.errorbar([1, 2, 3, 4, 5], gen2_val[1][0:5], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color=coherence_colors[1], label=gen, yerr=gen2_sem[1][0:5], capsize=4, capthick=1)
    ax2a2.errorbar([1, 2, 3], wt2_val[1][5:8], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color="black", label="wt", yerr=wt2_sem[1][5:8], capsize=4, capthick=1)
    ax2a2.errorbar([1, 2, 3], gen2_val[1][5:8], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color=coherence_colors[1], label=gen, yerr=gen2_sem[1][5:8], capsize=4, capthick=1)

    ax2b1.errorbar([1, 2, 3, 4, 5], wt2_val[2][0:5], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color="black", label="wt", yerr=wt2_sem[2][0:5], capsize=4, capthick=1)
    ax2b1.errorbar([1, 2, 3, 4, 5], gen2_val[2][0:5], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color=coherence_colors[2], label=gen, yerr=gen2_sem[2][0:5], capsize=4, capthick=1)
    ax2b2.errorbar([1, 2, 3], wt2_val[2][5:8], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color="black", label="wt", yerr=wt2_sem[2][5:8], capsize=4, capthick=1)
    ax2b2.errorbar([1, 2, 3], gen2_val[2][5:8], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color=coherence_colors[2], label=gen, yerr=gen2_sem[2][5:8], capsize=4, capthick=1)

    ax2c1.errorbar([1, 2, 3, 4, 5], wt2_val[3][0:5], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color="black", label="wt", yerr=wt2_sem[3][0:5], capsize=4, capthick=1)
    ax2c1.errorbar([1, 2, 3, 4, 5], gen2_val[3][0:5], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color=coherence_colors[3], label=gen, yerr=gen2_sem[3][0:5], capsize=4, capthick=1)
    ax2c2.errorbar([1, 2, 3], wt2_val[3][5:8], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color="black", label="wt", yerr=wt2_sem[3][5:8], capsize=4, capthick=1)
    ax2c2.errorbar([1, 2, 3], gen2_val[3][5:8], marker="o", markersize=5, markerfacecolor="white",
                   linewidth=1.5, color=coherence_colors[3], label=gen, yerr=gen2_sem[3][5:8], capsize=4, capthick=1)

    #################### FINISH ####################
    if printPDF:
        figPDF = plt.gcf()
        plt.draw()

        if fileIndex == 0:  # Access disc1_hetinx data
            figPDF.savefig('figs2a/disc1_hetinx_' + gen + '_Fig2a.pdf')
        elif fileIndex == 1:  # Access immp2l_NIBR data
            figPDF.savefig('figs2a/immp2l_NIBR_' + gen + '_Fig2a.pdf')
        elif fileIndex == 2:  # Access immp2l_summer data
            figPDF.savefig('figs2a/immp2l_summer_' + gen + '_Fig2a.pdf')
        elif fileIndex == 3:  # Access scn1lab_NIBR data
            figPDF.savefig('figs2a/scn1lab_NIBR_' + gen + '_Fig2a.pdf')
        elif fileIndex == 4:  # Access scn1lab_sa16474 data
            figPDF.savefig('figs2a/scn1lab_sa16474_' + gen + '_Fig2a.pdf')
        elif fileIndex == 5:  # Access surrogate_fish1 data
            figPDF.savefig('figs2a/surrogate_fish1_' + gen + '_Fig2a.pdf')
        elif fileIndex == 6:  # Access surrogate_fish2 data
            figPDF.savefig('figs2a/surrogate_fish2_' + gen + '_Fig2a.pdf')
        elif fileIndex == 7:  # Access surrogate_fish3 data
            figPDF.savefig('figs2a/surrogate_fish3_' + gen + '_Fig2a.pdf')
        elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
            figPDF.savefig('figs2a/scn1lab_NIBR_20200708_' + gen + '_Fig2a.pdf')
        elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
            figPDF.savefig('figs2a/scn1lab_zirc_20200710_' + gen + '_Fig2a.pdf')

    else:
        plt.show()

def p2b(gen, fileIndex, printPDF):
    #################### SET UP ####################
    fig = plt.figure(figsize=(4, 5))
    fig.subplots_adjust(bottom=0.15)
    outer = gridspec.GridSpec(2, 1, wspace=0.3, hspace=0.4)
    inner1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
    inner2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.1, hspace=0.1,
                                              width_ratios=(6.5, 4))
    ax1 = plt.Subplot(fig, inner1[0])
    ax2a = plt.Subplot(fig, inner2[0])
    ax2b = plt.Subplot(fig, inner2[1])
    subplot_list = [ax1, ax2a, ax2b]
    for subplt in subplot_list:
        fig.add_subplot(subplt)

    if fileIndex == 0:  # Access disc1_hetinx data
        fig.suptitle('\ndisc1_hetinx ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 1:  # Access immp2l_NIBR data
        fig.suptitle('\nimmp2l_NIBR ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 2:  # Access immp2l_summer data
        fig.suptitle('\nimmp2l_summer ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 3:  # Access scn1lab_NIBR data
        fig.suptitle('\nscn1lab_NIBR ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 4:  # Access scn1lab_sa16474 data
        fig.suptitle('\nscn1lab_sa16474 ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 5:  # Access surrogate_fish1 data
        fig.suptitle('\nsurrogate_fish1 ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 6:  # Access surrogate_fish2 data
        fig.suptitle('\nsurrogate_fish2 ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 7:  # Access surrogate_fish3 data
        fig.suptitle('\nsurrogate_fish3 ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
        fig.suptitle('\nscn1lab_NIBR_20200708 ' + gen + ' Fig2b', y=0.99)
    elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
        fig.suptitle('\nscn1lab_zirc_20200710 ' + gen + ' Fig2b', y=0.99)

    coherence = [0, 25, 50, 100]
    coherence_colors = ["black", "maroon", "firebrick", "indianred"]
    if gen == "het":
        gen_color = "purple"
    elif gen == "hom":
        gen_color = "darkblue"

    data = loadData(2)

    #################### PLOT 2B ####################
    # 2B: Time-binned probability correct (wt or gen)
    wt1 = np.asarray(getData("twoB", "wt", fileIndex, data)[1])
    gen1 = np.asarray(getData("twoB", gen, fileIndex, data)[1])

    wt1_val = wt1[0]
    gen1_val = gen1[0]
    wt1_sem = wt1[1]
    gen1_sem = gen1[1]

    ax1.set_ylim([40, 100])
    ax1.set_yticks([50, 70, 90])
    ax1.set_xlim([7, 25])
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylabel("Probability correct (%)")

    ax1.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    ax1.plot(np.linspace(20.25, 25), np.linspace(20.25, 25) * 0 + 40.5, color="black")
    ax1.text(22, 34, "5 s", fontsize=8)
    bracket(ax1, text="Stimulus period", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),
            linekw=dict(color="black", lw=1.5))

    ax1.errorbar(np.linspace(8, 24, num=len(wt1_val[2])), wt1_val[2], marker="o", markersize=5, markerfacecolor="white",
                 linewidth=1.5, color="black", label="wt 50", yerr=wt1_sem[2], capsize=4, capthick=1)
    ax1.errorbar(np.linspace(8, 24, num=len(gen1_val[2])), gen1_val[2], marker="o", markersize=5,
                 markerfacecolor="white",
                 linewidth=1.5, color=coherence_colors[2], label=gen + " 50", yerr=gen1_sem[2], capsize=4, capthick=1)

    ax1.errorbar(np.linspace(8, 24, num=len(wt1_val[3])), wt1_val[3], marker="o", markersize=5, markerfacecolor="white",
                 linewidth=1.5, color="gray", label="wt 100", yerr=wt1_sem[3], capsize=4, capthick=1)
    ax1.errorbar(np.linspace(8, 24, num=len(gen1_val[3])), gen1_val[3], marker="o", markersize=5,
                 markerfacecolor="white",
                 linewidth=1.5, color=coherence_colors[3], label=gen + " 100", yerr=gen1_sem[3], capsize=4, capthick=1)

    handles, labels = ax1.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax1.legend(handles, labels, loc="upper right", frameon=False, prop={'size': 6})

    #################### PLOT 2C ####################
    # 2C: Probability correct of consecutive bouts after stimulus start and end (wt or gen)
    wt2 = np.asarray(getData("twoC", "wt", fileIndex, data)[1])
    gen2 = np.asarray(getData("twoC", gen, fileIndex, data)[1])

    wt2_val = wt2[0]
    gen2_val = gen2[0]
    wt2_sem = wt2[1]
    gen2_sem = gen2[1]

    ax2a.set_ylim([40, 100])
    ax2a.set_yticks([50, 70, 90])
    ax2a.set_xlim([0.5, 5.5])
    ax2a.set_xticks([1, 2, 3, 4, 5])
    ax2a.spines["top"].set_visible(False)
    ax2a.spines["right"].set_visible(False)
    ax2a.add_patch(matplotlib.patches.Rectangle((0.5, 5.5), 5.5, 100, color=(0.8, 0.8, 0.8)))
    ax2a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2a.set_ylabel("Probability correct (%)")
    ax2a.set_xlabel("Bout no. since\nstimulus start", fontsize=8)

    ax2b.set_ylim([40, 100])
    ax2b.set_xlim([0.5, 3.5])
    ax2b.set_xticks([1, 2, 3])
    ax2b.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_visible(False)
    ax2b.spines["left"].set_visible(False)
    ax2b.add_patch(matplotlib.patches.Rectangle((0.3, 1), 0.5, 100, color=(0.8, 0.8, 0.8)))
    ax2b.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2b.set_xlabel("Bout no. since\nstimulus end", fontsize=8)

    ax2a.errorbar([1, 2, 3, 4, 5], wt2_val[2][0:5], marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color="black", label="wt", yerr=wt2_sem[2][0:5], capsize=4, capthick=1)
    ax2a.errorbar([1, 2, 3, 4, 5], gen2_val[2][0:5], marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color=coherence_colors[2], label=gen, yerr=gen2_sem[2][0:5], capsize=4, capthick=1)
    ax2b.errorbar([1, 2, 3], wt2_val[2][5:8], marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color="black", label="wt", yerr=wt2_sem[2][5:8], capsize=4, capthick=1)
    ax2b.errorbar([1, 2, 3], gen2_val[2][5:8], marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color=coherence_colors[2], label=gen, yerr=gen2_sem[2][5:8], capsize=4, capthick=1)

    ax2a.errorbar([1, 2, 3, 4, 5], wt2_val[3][0:5], marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color="gray", label="wt", yerr=wt2_sem[3][0:5], capsize=4, capthick=1)
    ax2a.errorbar([1, 2, 3, 4, 5], gen2_val[3][0:5], marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color=coherence_colors[3], label=gen, yerr=gen2_sem[3][0:5], capsize=4, capthick=1)
    ax2b.errorbar([1, 2, 3], wt2_val[3][5:8], marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color="gray", label="wt", yerr=wt2_sem[3][5:8], capsize=4, capthick=1)
    ax2b.errorbar([1, 2, 3], gen2_val[3][5:8], marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color=coherence_colors[3], label=gen, yerr=gen2_sem[3][5:8], capsize=4, capthick=1)

    #################### FINISH ####################
    if printPDF:
        figPDF = plt.gcf()
        plt.draw()

        if fileIndex == 0:  # Access disc1_hetinx data
            figPDF.savefig('figs2b/disc1_hetinx_' + gen + '_Fig2b.pdf')
        elif fileIndex == 1:  # Access immp2l_NIBR data
            figPDF.savefig('figs2b/immp2l_NIBR_' + gen + '_Fig2b.pdf')
        elif fileIndex == 2:  # Access immp2l_summer data
            figPDF.savefig('figs2b/immp2l_summer_' + gen + '_Fig2b.pdf')
        elif fileIndex == 3:  # Access scn1lab_NIBR data
            figPDF.savefig('figs2b/scn1lab_NIBR_' + gen + '_Fig2b.pdf')
        elif fileIndex == 4:  # Access scn1lab_sa16474 data
            figPDF.savefig('figs2b/scn1lab_sa16474_' + gen + '_Fig2b.pdf')
        elif fileIndex == 5:  # Access surrogate_fish1 data
            figPDF.savefig('figs2b/surrogate_fish1_' + gen + '_Fig2b.pdf')
        elif fileIndex == 6:  # Access surrogate_fish2 data
            figPDF.savefig('figs2b/surrogate_fish2_' + gen + '_Fig2b.pdf')
        elif fileIndex == 7:  # Access surrogate_fish3 data
            figPDF.savefig('figs2b/surrogate_fish3_' + gen + '_Fig2b.pdf')
        elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
            figPDF.savefig('figs2b/scn1lab_NIBR_20200708_' + gen + '_Fig2b.pdf')
        elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
            figPDF.savefig('figs2b/scn1lab_zirc_20200710_' + gen + '_Fig2b.pdf')

    else:
        plt.show()

def p2v(gen, fileIndex, printPDF):
    #################### SET UP ####################
    fig = plt.figure(figsize=(6, 9))
    fig.subplots_adjust()
    outer = gridspec.GridSpec(4, 1, hspace=0.45)
    inner1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.5, hspace=0.1)
    inner2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
    inner3 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[2], wspace=0.1, hspace=0.1, width_ratios=(6.5, 4, 0.5, 6.5, 4))
    inner4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[3], wspace=0.1, hspace=0.1)
    ax1a = plt.Subplot(fig, inner1[0])
    ax1b = plt.Subplot(fig, inner1[1])
    ax2a = plt.Subplot(fig, inner2[0])
    ax2b = plt.Subplot(fig, inner2[1])
    ax3a = plt.Subplot(fig, inner3[0])
    ax3b = plt.Subplot(fig, inner3[1])
    ax3c = plt.Subplot(fig, inner3[3])
    ax3d = plt.Subplot(fig, inner3[4])
    ax4 = plt.Subplot(fig, inner4[0])
    subplot_list = [ax1a, ax1b, ax2a, ax2b, ax3a, ax3b, ax3c, ax3d, ax4]
    for subplt in subplot_list:
        fig.add_subplot(subplt)

    if fileIndex == 0:  # Access disc1_hetinx data
        fig.suptitle('\ndisc1_hetinx ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 1:  # Access immp2l_NIBR data
        fig.suptitle('\nimmp2l_NIBR ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 2:  # Access immp2l_summer data
        fig.suptitle('\nimmp2l_summer ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 3:  # Access scn1lab_NIBR data
        fig.suptitle('\nscn1lab_NIBR ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 4:  # Access scn1lab_sa16474 data
        fig.suptitle('\nscn1lab_sa16474 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 5:  # Access surrogate_fish1 data
        fig.suptitle('\nsurrogate_fish1 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 6:  # Access surrogate_fish2 data
        fig.suptitle('\nsurrogate_fish2 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 7:  # Access surrogate_fish3 data
        fig.suptitle('\nsurrogate_fish3 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
        fig.suptitle('\nscn1lab_NIBR_20200708 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
        fig.suptitle('\nscn1lab_zirc_20200710 ' + gen + ' Fig2', y=0.96)

    coherence = [0, 25, 50, 100]
    if gen == "het":
        gen_color = "purple"
    elif gen == "hom":
        gen_color = "darkblue"

    data = loadData(2)

    #################### PLOT 2Aa ####################
    # 2Aa: Probability correct as a function of coherence (wt and gen)
    wt1a = np.asarray(getData("twoAa", "wt", fileIndex, data)[1])
    gen1a = np.asarray(getData("twoAa", gen, fileIndex, data)[1])
    wt1a_val = wt1a[0]
    gen1a_val = gen1a[0]
    wt1a_sem = wt1a[1]
    gen1a_sem = gen1a[1]

    ax1a.set_ylim([40, 100])
    ax1a.set_yticks([50, 70, 90])
    ax1a.set_xticks(coherence)
    ax1a.set(ylabel="Probability\ncorrect (%)", xlabel="Coherence (%)")
    ax1a.spines["top"].set_visible(False)
    ax1a.spines["right"].set_visible(False)

    ax1a.errorbar(coherence, wt1a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black", label="wt", yerr=wt1a_sem, capsize=4, capthick=1)
    ax1a.errorbar(coherence, gen1a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color, label=gen, yerr=gen1a_sem, capsize=4, capthick=1)
    ax1a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    #################### PLOT 2Ab ####################
    # 2Ab: Interbout interval as a function of coherence (wt and gen)
    wt1b = np.asarray(getData("twoAb", "wt", fileIndex, data)[1])
    gen1b = np.asarray(getData("twoAb", gen, fileIndex, data)[1])
    wt1b_val = wt1b[0]
    gen1b_val = gen1b[0]
    wt1b_sem = wt1b[1]
    gen1b_sem = gen1b[1]

    ax1b.set_ylim([0.25, 2.25])
    ax1b.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax1b.set_xticks(coherence)
    ax1b.set(ylabel="Inter bout\ninterval (s)", xlabel="Coherence (%)")
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["right"].set_visible(False)

    ax1b.errorbar(coherence, wt1b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                  label="wt", yerr=wt1b_sem, capsize=4, capthick=1)
    ax1b.errorbar(coherence, gen1b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5,
                  color=gen_color, label=gen, yerr=gen1b_sem, capsize=4, capthick=1)

    handles, labels = ax1b.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax1b.legend(handles, labels, loc="upper right", frameon=False, prop={'size': 8})

    #################### PLOT 2B ####################
    # 2B: Time-binned probability correct (wt or gen)
    wt2 = np.asarray(getData("twoB", "wt", fileIndex, data)[1])
    gen2 = np.asarray(getData("twoB", gen, fileIndex, data)[1])
    wt2_val = wt2[0]
    gen2_val = gen2[0]
    wt2_sem = wt2[1]
    gen2_sem = gen2[1]
    ax2_colors = ["black", "maroon", "firebrick", "indianred"]

    ax2a.set_ylim([40, 100])
    ax2a.set_yticks([50, 70, 90])
    ax2b.set_xlim([7, 25])
    ax2a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax2a.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax2a.spines["top"].set_visible(False)
    ax2a.spines["bottom"].set_visible(False)
    ax2a.spines["right"].set_visible(False)
    ax2a.set_ylabel("Probability correct (%)")
    ax2a.text(12.5, 25, "(bin = 2 s)")

    ax2b.set_ylim([40, 100])
    ax2b.set_yticks([50, 70, 90])
    ax2b.set_xlim([7, 25])
    ax2b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax2b.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["bottom"].set_visible(False)
    ax2b.spines["right"].set_visible(False)
    ax2b.text(12.5, 25, "(bin = 2 s)")

    for i in range(0, 4):
        ax2a.errorbar(np.linspace(8, 24, num=len(wt2_val[i])), wt2_val[i], marker="o", markersize=5,
                      markerfacecolor="white",
                      linewidth=1.5, color=ax2_colors[i], label="wt", yerr=wt2_sem[i], capsize=4, capthick=1)

    ax2a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2a.text(23, 94, "wt", fontsize=8)

    for i in range(0, 4):
        ax2b.errorbar(np.linspace(8, 24, num=len(gen2_val[i])), gen2_val[i], marker="o", markersize=5,
                      markerfacecolor="white", linewidth=1.5, color=ax2_colors[i], label=gen, yerr=gen2_sem[i],
                      capsize=4, capthick=1)

    ax2b.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax2b.text(23, 94, gen, fontsize=8)

    ax2a.plot(np.linspace(20.25, 25), np.linspace(20.25, 25) * 0 + 40.5, color="black")
    ax2a.text(22, 34, "5 s", fontsize=8)
    ax2b.plot(np.linspace(20.25, 25), np.linspace(20.25, 25) * 0 + 40.5, color="black")
    ax2b.text(22, 34, "5 s", fontsize=8)
    bracket(ax2a, text="Stimulus period", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),
            linekw=dict(color="black", lw=1.5))
    bracket(ax2b, text="Stimulus period", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),
            linekw=dict(color="black", lw=1.5))

    #################### PLOT 2C ####################
    # 2C: Probability correct of consecutive bouts after stimulus start and end (wt or gen)
    wt3 = np.asarray(getData("twoC", "wt", fileIndex, data)[1])
    gen3 = np.asarray(getData("twoC", gen, fileIndex, data)[1])
    wt3_val = wt3[0]
    gen3_val = gen3[0]
    wt3_sem = wt3[1]
    gen3_sem = gen3[1]

    ax3_colors = ["black", "maroon", "firebrick", "indianred"]

    ax3a.set_ylim([40, 100])
    ax3a.set_yticks([50, 70, 90])
    ax3a.set_xlim([0.5, 5.5])
    ax3a.set_xticks([1, 2, 3, 4, 5])
    ax3a.spines["top"].set_visible(False)
    ax3a.spines["right"].set_visible(False)
    ax3a.add_patch(matplotlib.patches.Rectangle((0.5, 5.5), 5.5, 100, color=(0.8, 0.8, 0.8)))
    ax3a.set_ylabel("Probability correct (%)")
    ax3a.set_xlabel("Bout no. since\nstimulus start", fontsize=8)
    ax3a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    ax3b.set_ylim([40, 100])
    ax3b.set_xlim([0.5, 3.5])
    ax3b.set_xticks([1, 2, 3])
    ax3b.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax3b.spines["top"].set_visible(False)
    ax3b.spines["right"].set_visible(False)
    ax3b.spines["left"].set_visible(False)
    ax3b.add_patch(matplotlib.patches.Rectangle((0.3, 1), 0.5, 100, color=(0.8, 0.8, 0.8)))
    ax3b.text(2.7, 94, "wt", fontsize=8)
    ax3b.set_xlabel("Bout no. since\nstimulus end", fontsize=8)
    ax3b.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    ax3c.set_ylim([40, 100])
    ax3c.set_yticks([50, 70, 90])
    ax3c.set_xlim([0.5, 5.5])
    ax3c.set_xticks([1, 2, 3, 4, 5])
    ax3c.spines["top"].set_visible(False)
    ax3c.spines["right"].set_visible(False)
    ax3c.add_patch(matplotlib.patches.Rectangle((0.5, 5.5), 5.5, 100, color=(0.8, 0.8, 0.8)))
    ax3c.set_xlabel("Bout no. since\nstimulus start", fontsize=8)
    ax3c.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    ax3d.set_ylim([40, 100])
    ax3d.set_xlim([0.5, 3.5])
    ax3d.set_xticks([1, 2, 3])
    ax3d.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax3d.spines["top"].set_visible(False)
    ax3d.spines["right"].set_visible(False)
    ax3d.spines["left"].set_visible(False)
    ax3d.add_patch(matplotlib.patches.Rectangle((0.3, 1), 0.5, 100, color=(0.8, 0.8, 0.8)))
    ax3d.text(2.7, 94, gen, fontsize=8)
    ax3d.set_xlabel("Bout no. since\nstimulus end", fontsize=8)
    ax3d.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)

    for i in range(0, 4):
        ax3a.errorbar([1, 2, 3, 4, 5], wt3_val[i][0:5], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax3_colors[i], label="wt", yerr=wt3_sem[i][0:5], capsize=4, capthick=1)
        ax3b.errorbar([1, 2, 3], wt3_val[i][5:8], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax3_colors[i], label="wt", yerr=wt3_sem[i][5:8], capsize=4, capthick=1)
        ax3c.errorbar([1, 2, 3, 4, 5], gen3_val[i][0:5], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax3_colors[i], label=gen, yerr=gen3_sem[i][0:5], capsize=4, capthick=1)
        ax3d.errorbar([1, 2, 3], gen3_val[i][5:8], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax3_colors[i], label=gen, yerr=gen3_sem[i][5:8], capsize=4, capthick=1)

    #################### PLOT 2D ####################
    # 2D: Probability to swim in same direction as a function of interbout interval (0% coherence, wt and gen)
    wt4 = np.asarray(getData("twoD", "wt", fileIndex, data)[1])
    gen4 = np.asarray(getData("twoD", gen, fileIndex, data)[1])
    wt4_val = wt4[0]
    gen4_val = gen4[0]
    wt4_sem = wt4[1]
    gen4_sem = gen4[1]

    ax4.set_ylim([40, 100])
    ax4.set_yticks([50, 70, 90])
    ax4.set_xlim([-0.4, 3.1])
    ax4.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_ylabel("Probability to swim in\nsame direction (%)", fontsize=9)
    ax4.set_xlabel("Inter bout interval at 0% coherence\n(bin = 0.5 s)", fontsize=9)

    ax4.errorbar(np.linspace(0, 3, len(wt4_val)), wt4_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black", label="wt", yerr=wt4_sem, capsize=4, capthick=1)
    ax4.errorbar(np.linspace(0, 3, len(gen4_val)), gen4_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color, label=gen, yerr=gen4_sem, capsize=4, capthick=1)
    ax4.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax4.axvline(x=-0.2, linestyle="dotted", color="black", alpha=0.5)

    handles, labels = ax4.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax4.legend(handles, labels, loc="upper right", frameon=False, prop={'size': 8})

    ax4.annotate('Time of\nlast bout', xy=(-0.2, 40), xycoords='data', xytext=(-0.2, 25), textcoords='data',
                  arrowprops=dict(arrowstyle='-|>', color='black', lw=2), fontsize=8, horizontalalignment="center")

    #################### FINISH ####################
    if printPDF:
        figPDF = plt.gcf()
        plt.draw()

        if fileIndex == 0:  # Access disc1_hetinx data
            figPDF.savefig('figs2v/disc1_hetinx_' + gen + '_Fig2v.pdf')
        elif fileIndex == 1:  # Access immp2l_NIBR data
            figPDF.savefig('figs2v/immp2l_NIBR_' + gen + '_Fig2v.pdf')
        elif fileIndex == 2:  # Access immp2l_summer data
            figPDF.savefig('figs2v/immp2l_summer_' + gen + '_Fig2v.pdf')
        elif fileIndex == 3:  # Access scn1lab_NIBR data
            figPDF.savefig('figs2v/scn1lab_NIBR_' + gen + '_Fig2v.pdf')
        elif fileIndex == 4:  # Access scn1lab_sa16474 data
            figPDF.savefig('figs2v/scn1lab_sa16474_' + gen + '_Fig2v.pdf')
        elif fileIndex == 5:  # Access surrogate_fish1 data
            figPDF.savefig('figs2v/surrogate_fish1_' + gen + '_Fig2v.pdf')
        elif fileIndex == 6:  # Access surrogate_fish2 data
            figPDF.savefig('figs2v/surrogate_fish2_' + gen + '_Fig2v.pdf')
        elif fileIndex == 7:  # Access surrogate_fish3 data
            figPDF.savefig('figs2v/surrogate_fish3_' + gen + '_Fig2v.pdf')
        elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
            figPDF.savefig('figs2v/scn1lab_NIBR_20200708_' + gen + '_Fig2v.pdf')
        elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
            figPDF.savefig('figs2v/scn1lab_zirc_20200710_' + gen + '_Fig2v.pdf')
    else:
        plt.show()

def p3(gen, fileIndex, printPDF):
    #################### SET UP ####################
    fig = plt.figure(figsize=(12, 3.5))
    fig.subplots_adjust(bottom=0.15)
    outer = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.2, height_ratios=(0.1, 9.9))
    inner1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[4], wspace=0.1, hspace=0.1)
    inner2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[5], wspace=0.1, hspace=0.1)
    inner3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[6], wspace=0.1, hspace=0.1)
    inner4 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[7], wspace=0.1, hspace=0.1)
    ax1a = plt.Subplot(fig, inner1[0])
    ax1b = plt.Subplot(fig, inner1[1])
    ax2a = plt.Subplot(fig, inner2[0])
    ax2b = plt.Subplot(fig, inner2[1])
    ax3a = plt.Subplot(fig, inner3[0])
    ax3b = plt.Subplot(fig, inner3[1])
    ax4a = plt.Subplot(fig, inner4[0])
    ax4b = plt.Subplot(fig, inner4[1])
    subplot_list = [ax1a, ax1b, ax2a, ax2b, ax3a, ax3b, ax4a, ax4b]
    for subplt in subplot_list:
        fig.add_subplot(subplt)

    if fileIndex == 0:  # Access disc1_hetinx data
        fig.suptitle('\ndisc1_hetinx ' + gen + ' Fig3', y=0.96)
    elif fileIndex == 1:  # Access immp2l_NIBR data
        fig.suptitle('\nimmp2l_NIBR ' + gen + ' Fig3', y=0.96)
    elif fileIndex == 2:  # Access immp2l_summer data
        fig.suptitle('\nimmp2l_summer ' + gen + ' Fig3', y=0.96)
    elif fileIndex == 3:  # Access scn1lab_NIBR data
        fig.suptitle('\nscn1lab_NIBR ' + gen + ' Fig3', y=0.96)
    elif fileIndex == 4:  # Access scn1lab_sa16474 data
        fig.suptitle('\nscn1lab_sa16474 ' + gen + ' Fig3', y=0.96)
    elif fileIndex == 5:  # Access surrogate_fish1 data
        fig.suptitle('\nsurrogate_fish1 ' + gen + ' Fig3', y=0.96)
    elif fileIndex == 6:  # Access surrogate_fish2 data
        fig.suptitle('\nsurrogate_fish2 ' + gen + ' Fig3', y=0.96)
    elif fileIndex == 7:  # Access surrogate_fish3 data
        fig.suptitle('\nsurrogate_fish3 ' + gen + ' Fig3', y=0.96)
    elif fileIndex == 8: # Access scn1lab_NIBR_20200708 data
        fig.suptitle('\nscn1lab_NIBR_20200708 ' + gen + ' Fig2', y=0.96)
    elif fileIndex == 9: # Access scn1lab_zirc_20200710 data
        fig.suptitle('\nscn1lab_zirc_20200710 ' + gen + ' Fig2', y=0.96)

    coherence = [0, 25, 50, 100]
    if gen == "het":
        gen_color = "purple"
    elif gen == "hom":
        gen_color = "darkblue"

    data = loadData(3)

    #################### PLOT 3A ####################
    # 3A Probability correct as a function of delay (interbout interval) for all stimulus levels (wt or gen)
    wt1 = np.asarray(getData("threeA", "wt", fileIndex, data)[1])
    gen1 = np.asarray(getData("threeA", gen, fileIndex, data)[1])
    wt1_val = wt1[0]
    wt1_sem = wt1[1]
    gen1_val = gen1[0]
    gen1_sem = gen1[1]

    ax1a.text(-1, 38, "Probability correct (%)", rotation="vertical", va="center")
    ax1_colors = ["black", "maroon", "firebrick", "indianred"]

    ax1a.set_ylim([40, 100])
    ax1a.set_yticks([50, 70, 90])
    ax1a.set_xlim([-0.4, 2.1])
    ax1a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1a.spines["top"].set_visible(False)
    ax1a.spines["bottom"].set_visible(False)
    ax1a.spines["right"].set_visible(False)

    ax1b.set_ylim([40, 100])
    ax1b.set_yticks([50, 70, 90])
    ax1b.set_xlim([-0.4, 2.1])
    ax1b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["bottom"].set_visible(False)
    ax1b.spines["right"].set_visible(False)
    ax1b.set_xlabel("Inter bout\ninterval\n(bin = 0.5 s)")

    for i in range(0, 4):
        ax1a.errorbar(np.linspace(0, 2, num=len(wt1_val[i])), wt1_val[i], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax1_colors[i], label="wt", yerr=wt1_sem[i], capsize=4, capthick=1)
        ax1b.errorbar(np.linspace(0, 2, num=len(gen1_val[i])), gen1_val[i], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax1_colors[i], label=gen, yerr=gen1_sem[i], capsize=4, capthick=1)
    ax1a.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax1a.axvline(x=-0.2, linestyle="dotted", color="black", alpha=0.5)
    ax1b.axhline(y=50, linestyle="dotted", color="black", alpha=0.5)
    ax1b.axvline(x=-0.2, linestyle="dotted", color="black", alpha=0.5)

    ax1a.text(1.9, 94, "wt", fontsize=8)
    ax1b.text(1.9, 94, gen, fontsize=8)
    ax1b.annotate('Time of\nlast bout', xy=(-0.2, 40), xycoords='data', xytext=(-0.2, 22), textcoords='data',
                  arrowprops=dict(arrowstyle='-|>', color='black', lw=2), fontsize=8, horizontalalignment="center")

    #################### PLOT 3BC ####################
    wt23 = getData("threeBC", "wt", fileIndex, data)
    gen23 = getData("threeBC", gen, fileIndex, data)
    wt2_val = np.asarray(wt23[1][0])
    wt2_sem = np.asarray(wt23[1][1])
    gen2_val = np.asarray(gen23[1][0])
    gen2_sem = np.asarray(gen23[1][1])
    wt3_val = np.asarray(wt23[2][0])
    wt3_sem = np.asarray(wt23[2][1])
    gen3_val = np.asarray(gen23[2][0])
    gen3_sem = np.asarray(gen23[2][1])

    ax2a.text(-0.95, -3.3, "Correct bout turn angle amplitude (\N{DEGREE SIGN})", rotation="vertical", va="center", fontsize=9)
    ax3a.text(-0.95, -3.3, "Incorrect bout turn angle amplitude (\N{DEGREE SIGN})", rotation="vertical", va="center",
              fontsize=9)
    ax23_colors = ["black", "maroon", "firebrick", "indianred"]


    ax2a.set_ylim([0, 50])
    ax2a.set_yticks(range(0,55,20))
    ax2a.set_xlim([-0.4, 2.1])
    ax2a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax2a.spines["top"].set_visible(False)
    ax2a.spines["bottom"].set_visible(False)
    ax2a.spines["right"].set_visible(False)

    ax2b.set_ylim([0, 50])
    ax2b.set_yticks(range(0, 50, 20))
    ax2b.set_xlim([-0.4, 2.1])
    ax2b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["bottom"].set_visible(False)
    ax2b.spines["right"].set_visible(False)
    ax2b.set_xlabel("Inter bout\ninterval\n(bin = 0.5 s)")

    ax3a.set_ylim([0, 50])
    ax3a.set_yticks(range(0, 55, 20))
    ax3a.set_xlim([-0.4, 2.1])
    ax3a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax3a.spines["top"].set_visible(False)
    ax3a.spines["bottom"].set_visible(False)
    ax3a.spines["right"].set_visible(False)

    ax3b.set_ylim([0, 50])
    ax3b.set_yticks(range(0, 50, 20))
    ax3b.set_xlim([-0.4, 2.1])
    ax3b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax3b.spines["top"].set_visible(False)
    ax3b.spines["bottom"].set_visible(False)
    ax3b.spines["right"].set_visible(False)
    ax3b.set_xlabel("Inter bout\ninterval\n(bin = 0.5 s)")

    for i in range(0, 4):
        ax2a.errorbar(np.linspace(0, 2, num=len(wt2_val[i])), wt2_val[i], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax23_colors[i], label="wt", yerr=wt2_sem[i], capsize=4, capthick=1)
        ax2b.errorbar(np.linspace(0, 2, num=len(gen2_val[i])), gen2_val[i], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax23_colors[i], label=gen, yerr=gen2_sem[i], capsize=4, capthick=1)
        ax3a.errorbar(np.linspace(0, 2, num=len(wt3_val[i])), wt3_val[i], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax23_colors[i], label="wt", yerr=wt3_sem[i], capsize=4, capthick=1)
        ax3b.errorbar(np.linspace(0, 2, num=len(gen3_val[i])), gen3_val[i], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax23_colors[i], label=gen, yerr=gen3_sem[i], capsize=4, capthick=1)

    ax2a.axvline(x=-0.2, linestyle="dotted", color="black", alpha=0.5)
    ax2a.text(1.9, 47, "wt", fontsize=8)
    ax2b.axvline(x=-0.2, linestyle="dotted", color="black", alpha=0.5)
    ax2b.text(1.9, 47, gen, fontsize=8)
    ax3a.axvline(x=-0.2, linestyle="dotted", color="black", alpha=0.5)
    ax3a.text(1.9, 47, "wt", fontsize=8)
    ax3b.axvline(x=-0.2, linestyle="dotted", color="black", alpha=0.5)
    ax3b.text(1.9, 47, gen, fontsize=8)

    ax2b.annotate('Time of\nlast bout', xy=(-0.2, 0), xycoords='data', xytext=(-0.2, -15), textcoords='data',
                  arrowprops=dict(arrowstyle='-|>', color='black', lw=2), fontsize=8, horizontalalignment="center")
    ax3b.annotate('Time of\nlast bout', xy=(-0.2, 0), xycoords='data', xytext=(-0.2, -15), textcoords='data',
                  arrowprops=dict(arrowstyle='-|>', color='black', lw=2), fontsize=8, horizontalalignment="center")

    #################### PLOT 3D ####################
    # 3Dab Time for first bout and first correct bout after start of stimulus (wt and gen)
    wt4 = getData("threeD", "wt", fileIndex, data)
    gen4 = getData("threeD", gen, fileIndex, data)
    wt4a_val = np.asarray(wt4[1][0])
    wt4a_sem = np.asarray(wt4[1][1])
    gen4a_val = np.asarray(gen4[1][0])
    gen4a_sem = np.asarray(gen4[1][1])
    wt4b_val = np.asarray(wt4[2][0])
    wt4b_sem = np.asarray(wt4[2][1])
    gen4b_val = np.asarray(gen4[2][0])
    gen4b_sem = np.asarray(gen4[2][1])

    ax4a.set_ylim([0, 4])
    ax4a.set_yticks([1, 2, 3, 4])
    ax4a.set_xticks(coherence)
    ax4a.tick_params(labelbottom=False)
    ax4a.spines["top"].set_visible(False)
    ax4a.spines["right"].set_visible(False)
    ax4a.set_ylabel("Time of first bout \nafter stimulus start (s)", fontsize=7)

    ax4b.set_ylim([0, 4])
    ax4b.set_yticks([1, 2, 3, 4])
    ax4b.set_xticks(coherence)
    ax4b.spines["top"].set_visible(False)
    ax4b.spines["right"].set_visible(False)
    ax4b.set_xlabel("Coherence (%)")
    ax4b.set_ylabel("Time of first correct bout \nafter stimulus start (s)", fontsize=7)

    ax4a.errorbar(coherence, wt4a_val, marker="o", markersize=5, markerfacecolor="white",
                 linewidth=1.5, color="black", label="wt", yerr=wt4a_sem, capsize=4, capthick=1)
    ax4a.errorbar(coherence, gen4a_val, marker="o", markersize=5, markerfacecolor="white",
                 linewidth=1.5, color=gen_color, label=gen, yerr=gen4a_sem, capsize=4, capthick=1)
    ax4b.errorbar(coherence, wt4b_val, marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color="black", label="wt", yerr=wt4b_sem, capsize=4, capthick=1)
    ax4b.errorbar(coherence, gen4b_val, marker="o", markersize=5, markerfacecolor="white",
                  linewidth=1.5, color=gen_color, label=gen, yerr=gen4b_sem, capsize=4, capthick=1)
    ax4a.legend(loc="upper left", frameon=False, prop={'size': 8})

    handles, labels = ax4a.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax4a.legend(handles, labels, loc="upper right", frameon=False, prop={'size': 8})

    #################### FINISH ####################
    if printPDF:
        figPDF = plt.gcf()
        plt.draw()

        if fileIndex == 0:  # Access disc1_hetinx data
            figPDF.savefig('figs3/disc1_hetinx_' + gen + '_Fig3.pdf')
        elif fileIndex == 1:  # Access immp2l_NIBR data
            figPDF.savefig('figs3/immp2l_NIBR_' + gen + '_Fig3.pdf')
        elif fileIndex == 2:  # Access immp2l_summer data
            figPDF.savefig('figs3/immp2l_summer_' + gen + '_Fig3.pdf')
        elif fileIndex == 3:  # Access scn1lab_NIBR data
            figPDF.savefig('figs3/scn1lab_NIBR_' + gen + '_Fig3.pdf')
        elif fileIndex == 4:  # Access scn1lab_sa16474 data
            figPDF.savefig('figs3/scn1lab_sa16474_' + gen + '_Fig3.pdf')
        elif fileIndex == 5:  # Access surrogate_fish1 data
            figPDF.savefig('figs3/surrogate_fish1_' + gen + '_Fig3.pdf')
        elif fileIndex == 6:  # Access surrogate_fish2 data
            figPDF.savefig('figs3/surrogate_fish2_' + gen + '_Fig3.pdf')
        elif fileIndex == 7:  # Access surrogate_fish3 data
            figPDF.savefig('figs3/surrogate_fish3_' + gen + '_Fig3.pdf')
        elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
            figPDF.savefig('figs3/scn1lab_NIBR_20200708_'+gen+'_Fig3.pdf')
        elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
            figPDF.savefig('figs3/scn1lab_zirc_20200710_'+gen+'_Fig3.pdf')

    else:
        plt.show()

def p4(gen, fileIndex, printPDF):
    #################### SET UP ####################
    fig = plt.figure(figsize=(12, 3.5))
    fig.subplots_adjust(bottom=0.15)
    outer = gridspec.GridSpec(2, 5, wspace=0.55, hspace=0.2, height_ratios=(0.1, 9.9))
    inner1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[5], wspace=0.1, hspace=0.1)
    inner2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[6], wspace=0.1, hspace=0.1)
    inner3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[7], wspace=0.1, hspace=0.1)
    inner4 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[8], wspace=0.1, hspace=0.1)
    inner5 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[9], wspace=0.1, hspace=0.1)
    ax1a = plt.Subplot(fig, inner1[0])
    ax1b = plt.Subplot(fig, inner1[1])
    ax1c = plt.Subplot(fig, inner1[2])
    ax2a = plt.Subplot(fig, inner2[0])
    ax2b = plt.Subplot(fig, inner2[1])
    ax3a = plt.Subplot(fig, inner3[0])
    ax3b = plt.Subplot(fig, inner3[1])
    ax4a = plt.Subplot(fig, inner4[0])
    ax4b = plt.Subplot(fig, inner4[1])
    ax5a = plt.Subplot(fig, inner5[0])
    ax5b = plt.Subplot(fig, inner5[1])
    subplot_list = [ax1a, ax1b, ax1c, ax2a, ax2b, ax3a, ax3b, ax4a, ax4b, ax5a, ax5b]
    for subplt in subplot_list:
        fig.add_subplot(subplt)

    if fileIndex == 0:  # Access disc1_hetinx data
        fig.suptitle('\ndisc1_hetinx ' + gen + ' Fig4', y=0.9999)
    elif fileIndex == 1:  # Access immp2l_NIBR data
        fig.suptitle('\nimmp2l_NIBR ' + gen + ' Fig1', y=0.9999)
    elif fileIndex == 2:  # Access immp2l_summer data
        fig.suptitle('\nimmp2l_summer ' + gen + ' Fig4', y=0.9999)
    elif fileIndex == 3:  # Access scn1lab_NIBR data
        fig.suptitle('\nscn1lab_NIBR ' + gen + ' Fig4', y=0.9999)
    elif fileIndex == 4:  # Access scn1lab_sa16474 data
        fig.suptitle('\nscn1lab_sa16474 ' + gen + ' Fig4', y=0.9999)
    elif fileIndex == 5:  # Access surrogate_fish1 data
        fig.suptitle('\nsurrogate_fish1 ' + gen + ' Fig4', y=0.9999)
    elif fileIndex == 6:  # Access surrogate_fish2 data
        fig.suptitle('\nsurrogate_fish2 ' + gen + ' Fig4', y=0.9999)
    elif fileIndex == 7:  # Access surrogate_fish3 data
        fig.suptitle('\nsurrogate_fish3 ' + gen + ' Fig4', y=0.9999)
    elif fileIndex == 8: # Access scn1lab_NIBR_20200708 data
        fig.suptitle('\nscn1lab_NIBR_20200708 ' + gen + ' Fig2', y=0.9999)
    elif fileIndex == 9: # Access scn1lab_zirc_20200710 data
        fig.suptitle('\nscn1lab_zirc_20200710 ' + gen + ' Fig2', y=0.9999)

    coherence = [0, 25, 50, 100]
    if gen == "het":
        gen_color = "purple"
    elif gen == "hom":
        gen_color = "darkblue"

    data = loadData(4)

    #################### PLOT 4A ####################
    wt123 = getData("fourABC", "wt", fileIndex, data)
    gen123 = getData("fourABC", gen, fileIndex, data)

    wt1a_val = np.asarray(wt123[1][0])
    wt1a_sem = np.asarray(wt123[1][1])
    gen1a_val = np.asarray(gen123[1][0])
    gen1a_sem = np.asarray(gen123[1][1])
    wt1b_val = np.asarray(wt123[2][0])
    wt1b_sem = np.asarray(wt123[2][1])
    gen1b_val = np.asarray(gen123[2][0])
    gen1b_sem = np.asarray(gen123[2][1])
    wt1c_val = np.asarray(wt123[3][0])
    wt1c_sem = np.asarray(wt123[3][1])
    gen1c_val = np.asarray(gen123[3][0])
    gen1c_sem = np.asarray(gen123[3][1])

    ax1a.set_title("Average distance\nswam in each bout", fontsize=8)
    ax1a.set_ylim([0, 0.5])
    ax1a.set_yticks([0.05, 0.25, 0.45])
    ax1a.tick_params(axis='x', which='both', labelbottom=False)
    ax1a.spines["top"].set_visible(False)
    ax1a.spines["right"].set_visible(False)
    ax1a.set_xticks(coherence)
    ax1a.set_ylabel("All\nbouts (cm)", fontsize=7.5)

    ax1b.set_ylim([0, 0.5])
    ax1b.set_yticks([0.05, 0.25, 0.45])
    ax1b.tick_params(axis='x', which='both', labelbottom=False)
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["right"].set_visible(False)
    ax1b.set_xticks(coherence)
    ax1b.set_ylabel("Correct\nbouts (cm)", fontsize=7.5)

    ax1c.set_ylim([0, 0.5])
    ax1c.set_yticks([0.05, 0.25, 0.45])
    ax1c.spines["top"].set_visible(False)
    ax1c.spines["right"].set_visible(False)
    ax1c.set_xticks(coherence)
    ax1c.set_ylabel("Incorrect\nbouts (cm)", fontsize=7.5)
    ax1c.set_xlabel("Coherence (%)")

    ax1a.errorbar(coherence, wt1a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                 label="wt", yerr=wt1a_sem, capsize=4, capthick=1)
    ax1a.errorbar(coherence, gen1a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color,
                 label=gen, yerr=gen1a_sem, capsize=4, capthick=1)
    ax1b.errorbar(coherence, wt1b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                  label="wt", yerr=wt1b_sem, capsize=4, capthick=1)
    ax1b.errorbar(coherence, gen1b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color,
                  label=gen, yerr=gen1b_sem, capsize=4, capthick=1)
    ax1c.errorbar(coherence, wt1c_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                  label="wt", yerr=wt1c_sem, capsize=4, capthick=1)
    ax1c.errorbar(coherence, gen1c_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color=gen_color,
                  label=gen, yerr=gen1c_sem, capsize=4, capthick=1)

    #################### PLOT 4B ####################
    wt2a_val = np.asarray(wt123[4][0])
    wt2a_sem = np.asarray(wt123[4][1])
    gen2a_val = np.asarray(gen123[4][0])
    gen2a_sem = np.asarray(gen123[4][1])
    wt2b_val = np.asarray(wt123[5][0])
    wt2b_sem = np.asarray(wt123[5][1])
    gen2b_val = np.asarray(gen123[5][0])
    gen2b_sem = np.asarray(gen123[5][1])

    ax2a.set_title("Average rightward/leftward\ndistance swam in each bout", fontsize=8)
    ax2a.set_ylim([0, 0.5])
    ax2a.set_yticks([0.05, 0.25, 0.45])
    ax2a.tick_params(axis='x', which='both', labelbottom=False)
    ax2a.spines["top"].set_visible(False)
    ax2a.spines["right"].set_visible(False)
    ax2a.set_xticks(coherence)
    ax2a.set_ylabel("Correct bouts,\ndistance rightward (cm)", fontsize=7.5)

    ax2b.set_ylim([0, 0.5])
    ax2b.set_yticks([0.05, 0.25, 0.45])
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_visible(False)
    ax2b.set_xticks(coherence)
    ax2b.set_ylabel("Incorrect bouts,\ndistance leftward (cm)", fontsize=7.5)
    ax2b.set_xlabel("Coherence (%)")

    ax2a.errorbar(coherence, wt2a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                  label="wt", yerr=wt2a_sem, capsize=4, capthick=1)
    ax2a.errorbar(coherence, gen2a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5,
                  color=gen_color, label=gen, yerr=gen2a_sem, capsize=4, capthick=1)
    ax2b.errorbar(coherence, wt2b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                  label="wt", yerr=wt2b_sem, capsize=4, capthick=1)
    ax2b.errorbar(coherence, gen2b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5,
                  color=gen_color, label=gen, yerr=gen2b_sem, capsize=4, capthick=1)

    #################### PLOT 4C ####################
    wt3a_val = np.asarray(wt123[6][0])
    wt3a_sem = np.asarray(wt123[6][1])
    gen3a_val = np.asarray(gen123[6][0])
    gen3a_sem = np.asarray(gen123[6][1])
    wt3b_val = np.asarray(wt123[7][0])
    wt3b_sem = np.asarray(wt123[7][1])
    gen3b_val = np.asarray(gen123[7][0])
    gen3b_sem = np.asarray(gen123[7][1])

    ax3a.set_title("Average forward distance\nswam in each bout", fontsize=8)
    ax3a.set_ylim([0, 0.5])
    ax3a.set_yticks([0.05, 0.25, 0.45])
    ax3a.tick_params(axis='x', which='both', labelbottom=False)
    ax3a.spines["top"].set_visible(False)
    ax3a.spines["right"].set_visible(False)
    ax3a.set_xticks(coherence)
    ax3a.set_ylabel("Correct bouts,\ndistance forward (cm)", fontsize=7.5)

    ax3b.set_ylim([0, 0.5])
    ax3b.set_yticks([0.05, 0.25, 0.45])
    ax3b.spines["top"].set_visible(False)
    ax3b.spines["right"].set_visible(False)
    ax3b.set_xticks(coherence)
    ax3b.set_ylabel("Incorrect bouts,\ndistance forward (cm)", fontsize=7.5)
    ax3b.set_xlabel("Coherence (%)")

    ax3a.errorbar(coherence, wt3a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                  label="wt", yerr=wt3a_sem, capsize=4, capthick=1)
    ax3a.errorbar(coherence, gen3a_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5,
                  color=gen_color, label=gen, yerr=gen3a_sem, capsize=4, capthick=1)
    ax3b.errorbar(coherence, wt3b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5, color="black",
                  label="wt", yerr=wt3b_sem, capsize=4, capthick=1)
    ax3b.errorbar(coherence, gen3b_val, marker="o", markersize=5, markerfacecolor="white", linewidth=1.5,
                  color=gen_color, label=gen, yerr=gen3b_sem, capsize=4, capthick=1)

    handles, labels = ax3a.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax3a.legend(handles, labels, loc="upper right", frameon=False, prop={'size': 8})

    #################### PLOT 4D ####################
    wt45 = getData("fourDE", "wt", fileIndex, data)

    wt4a_val = np.asarray(wt45[1][0])
    wt4a_sem = np.asarray(wt45[1][1])
    wt4b_val = np.asarray(wt45[2][0])
    wt4b_sem = np.asarray(wt45[2][1])

    ax4_colors = ["black", "maroon", "firebrick", "indianred"]

    ax4a.set_ylim([0, 0.5])
    ax4a.set_yticks([0.05, 0.25, 0.45])
    ax4a.set_xlim([3, 27])
    ax4a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax4a.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax4a.spines["top"].set_visible(False)
    ax4a.spines["bottom"].set_visible(False)
    ax4a.spines["right"].set_visible(False)
    ax4a.set_ylabel("Correct\nbouts (cm)", fontsize=7.5)
    ax4a.text(8.5, 0.55, "Time-binned average distance swam in each bout", fontsize=8)
    ax4a.text(23, 0.45, "wt", fontsize=8)

    ax4b.set_ylim([0, 0.5])
    ax4b.set_yticks([0.05, 0.25, 0.45])
    ax4b.set_xlim([3, 27])
    ax4b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax4b.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax4b.spines["top"].set_visible(False)
    ax4b.spines["bottom"].set_visible(False)
    ax4b.spines["right"].set_visible(False)
    ax4b.set_ylabel("Incorrect\nbouts (cm)", fontsize=7.5)
    ax4b.text(23, 0.45, "wt", fontsize=8)
    ax4b.text(9, -0.18, "(bin = 5 s)")

    bracket(ax4b, text="Stimulus\nperiod", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),
            linekw=dict(color="black", lw=1.5))
    ax4b.plot(np.linspace(20.5, 25), np.linspace(20.5, 25) * 0 + 0.01, color="black")
    ax4b.text(21.25, -0.05, "5 s", fontsize=8)

    for i in range(0, 4):
        ax4a.errorbar(np.linspace(5, 25, num=len(wt4a_val[i])), wt4a_val[i], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax4_colors[i], label="wt", yerr=wt4a_sem[i], capsize=4, capthick=1)
        ax4b.errorbar(np.linspace(5, 25, num=len(wt4b_val[i])), wt4b_val[i], marker="o", markersize=5,
                      markerfacecolor="white", linewidth=1.5, color=ax4_colors[i], label=gen, yerr=wt4b_sem[i], capsize=4, capthick=1)

    #################### PLOT 4E ####################
    gen45 = getData("fourDE", gen, fileIndex, data)

    gen5a_val = np.asarray(gen45[1][0])
    gen5a_sem = np.asarray(gen45[1][1])
    gen5b_val = np.asarray(gen45[2][0])
    gen5b_sem = np.asarray(gen45[2][1])

    ax5a.set_ylim([0, 0.5])
    ax5a.set_yticks([0.05, 0.25, 0.45])
    ax5a.set_xlim([3, 27])
    ax5a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax5a.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax5a.spines["top"].set_visible(False)
    ax5a.spines["bottom"].set_visible(False)
    ax5a.spines["right"].set_visible(False)
    ax5a.set_ylabel("Correct\nbouts (cm)", fontsize=7.5)
    ax5a.text(23, 0.45, gen, fontsize=8)

    ax5b.set_ylim([0, 0.5])
    ax5b.set_yticks([0.05, 0.25, 0.45])
    ax5b.set_xlim([3, 27])
    ax5b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax5b.add_patch(matplotlib.patches.Rectangle((10, 0), 10, 100, color=(0.8, 0.8, 0.8)))
    ax5b.spines["top"].set_visible(False)
    ax5b.spines["bottom"].set_visible(False)
    ax5b.spines["right"].set_visible(False)
    ax5b.set_ylabel("Incorrect\nbouts (cm)", fontsize=7.5)
    ax5b.text(23, 0.45, gen, fontsize=8)
    ax5b.text(9, -0.18, "(bin = 5 s)")


    bracket(ax5b, text="Stimulus\nperiod", pos=[10, -0.03], scalex=10, textkw=dict(fontsize=8),
            linekw=dict(color="black", lw=1.5))
    ax5b.plot(np.linspace(20.5, 25), np.linspace(20.5, 25) * 0 + 0.01, color="black")
    ax5b.text(21.25, -0.05, "5 s", fontsize=8)

    for i in range(0, 4):
        ax5a.errorbar(np.linspace(5, 25, num=len(gen5a_val[i])), gen5a_val[i], marker="o", markersize=5, markerfacecolor="white",
                      linewidth=1.5, color=ax4_colors[i], label="wt", yerr=gen5a_sem[i], capsize=4, capthick=1)
        ax5b.errorbar(np.linspace(5, 25, num=len(gen5b_val[i])), gen5b_val[i], marker="o", markersize=5,
                      markerfacecolor="white", linewidth=1.5, color=ax4_colors[i], label=gen, yerr=gen5b_sem[i], capsize=4, capthick=1)

    #################### FINISH ####################
    if printPDF:
        figPDF = plt.gcf()
        plt.draw()

        if fileIndex == 0:  # Access disc1_hetinx data
            figPDF.savefig('figs4/disc1_hetinx_' + gen + '_Fig4.pdf')
        elif fileIndex == 1:  # Access immp2l_NIBR data
            figPDF.savefig('figs4/immp2l_NIBR_' + gen + '_Fig4.pdf')
        elif fileIndex == 2:  # Access immp2l_summer data
            figPDF.savefig('figs4/immp2l_summer_' + gen + '_Fig4.pdf')
        elif fileIndex == 3:  # Access scn1lab_NIBR data
            figPDF.savefig('figs4/scn1lab_NIBR_' + gen + '_Fig4.pdf')
        elif fileIndex == 4:  # Access scn1lab_sa16474 data
            figPDF.savefig('figs4/scn1lab_sa16474_' + gen + '_Fig4.pdf')
        elif fileIndex == 5:  # Access surrogate_fish1 data
            figPDF.savefig('figs4/surrogate_fish1_' + gen + '_Fig4.pdf')
        elif fileIndex == 6:  # Access surrogate_fish2 data
            figPDF.savefig('figs4/surrogate_fish2_' + gen + '_Fig4.pdf')
        elif fileIndex == 7:  # Access surrogate_fish3 data
            figPDF.savefig('figs4/surrogate_fish3_' + gen + '_Fig4.pdf')
        elif fileIndex == 8:  # Access scn1lab_NIBR_20200708 data
            figPDF.savefig('figs4/scn1lab_NIBR_20200708_' + gen + '_Fig4.pdf')
        elif fileIndex == 9:  # Access scn1lab_zirc_20200710 data
            figPDF.savefig('figs4/scn1lab_zirc_20200710_' + gen + '_Fig4.pdf')

    else:
        plt.show()


#############################################################################
################################### MISC ####################################
#############################################################################
def loadData(figIndex):
    with open('analysis_output/analysis_output'+str(figIndex)+'.txt') as json_file:
        data = json.load(json_file)
    return(data)

def getData(funcName, gen, fileIndex, data):
    for n in data[funcName]:
        if n["gen"] == gen and n["fileIndex"] == fileIndex:
            return n["output"]

def bracket(ax, pos=[0,0], scalex=1, scaley=1, text="",textkw = {}, linekw = {}):
    x = np.array([0, 0.05, 0.45,0.5])
    y = np.array([0,-0.01,-0.01,-0.02])
    x = np.concatenate((x,x+0.5))
    y = np.concatenate((y,y[::-1]))
    ax.plot(x*scalex+pos[0], y*scaley+pos[1], clip_on=False,
            transform=ax.get_xaxis_transform(), **linekw)
    ax.text(pos[0]+0.5*scalex, (y.min()-0.01)*scaley+pos[1], text,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", **textkw)