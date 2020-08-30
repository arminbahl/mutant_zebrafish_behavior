import pylab as pl
import pandas as pd
from pathlib import Path
import numpy as np

root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence/chrna2a")

df_extracted_features = pd.read_hdf(root_path / "all_data.h5", key="extracted_features")
df_extracted_binned_features = pd.read_hdf(root_path / "all_data.h5", key="extracted_binned_features")
df_extracted_binned_features_same_direction = pd.read_hdf(root_path / "all_data.h5", key="extracted_binned_features_same_direction")
df_extracted_binned_features_heading_angle_change_histograms = pd.read_hdf(root_path / "all_data.h5", key="extracted_binned_features_heading_angle_change_histograms")
df_extracted_binned_features_inter_bout_interval_histograms = pd.read_hdf(root_path / "all_data.h5", key="extracted_binned_features_inter_bout_interval_histograms")

result1 = df_extracted_binned_features_heading_angle_change_histograms.query("genotype == 'wt'").groupby(["stim", "bin"]).mean()
result2 = df_extracted_binned_features_inter_bout_interval_histograms.query("genotype == 'wt'").groupby(["stim", "bin"]).mean()

# pl.plot(result1.loc[0])
# pl.plot(result1.loc[1])
# pl.plot(result1.loc[2])
# pl.plot(result1.loc[3])
# pl.figure()
#
# pl.plot(result2.loc[0])
# pl.plot(result2.loc[1])
# pl.plot(result2.loc[2])
# pl.plot(result2.loc[3])
# pl.show()
#
# pl.show()
#
# print(result)
# df
# genotypes = ["scn1lab+/+", "scn1lab+/-", "scn1lab-/-"]
# colors = ["black", "blue", "red"]

genotypes = ["wt", "het", "hom"]
colors = ["black", "blue", "red"]

for genotype, color in zip(genotypes, colors):
    #print(df_extracted_features)
    #print(genotype)
    correctness_as_function_of_coherence_mean = df_extracted_features.query("genotype == @genotype").groupby("stim").mean()["correctness"]
    correctness_as_function_of_coherence_sem = df_extracted_features.query("genotype == @genotype").groupby("stim").sem()["correctness"]

    inter_bout_interval_as_function_of_coherence_mean = df_extracted_features.query("genotype == @genotype").groupby("stim").mean()["inter_bout_interval"]
    inter_bout_interval_as_function_of_coherence_sem = df_extracted_features.query("genotype == @genotype").groupby("stim").sem()["inter_bout_interval"]

    binned_correctness_mean = df_extracted_binned_features.query("genotype == @genotype").groupby(["stim", "bin"]).mean()
    binned_correctness_sem = df_extracted_binned_features.query("genotype == @genotype").groupby(["stim", "bin"]).sem()

    binned_same_direction_mean = df_extracted_binned_features_same_direction.query("genotype == @genotype").groupby(["bin"]).mean()
    binned_same_direction_sem = df_extracted_binned_features_same_direction.query("genotype == @genotype").groupby(["bin"]).sem()

    if (len(binned_same_direction_mean)) == 0:
        continue

    pl.subplot(2, 2, 1)
    pl.plot(correctness_as_function_of_coherence_mean.index, correctness_as_function_of_coherence_mean.values, color=color)
    pl.errorbar(correctness_as_function_of_coherence_mean.index, correctness_as_function_of_coherence_mean.values, correctness_as_function_of_coherence_sem.values, color=color)

    pl.subplot(2, 2, 2)
    pl.plot(inter_bout_interval_as_function_of_coherence_mean.index, inter_bout_interval_as_function_of_coherence_mean.values, color=color)
    pl.errorbar(inter_bout_interval_as_function_of_coherence_mean.index, inter_bout_interval_as_function_of_coherence_mean.values, inter_bout_interval_as_function_of_coherence_sem.values, color=color)

    pl.subplot(2, 2, 3)
    for i in range(4):
        print("sdfsdf")

        print(binned_correctness_mean.loc[i, :].index)

        print("skljflgsjjdf3")
        print(binned_correctness_mean.loc[i, :].values)

        pl.plot(binned_correctness_mean.loc[i, :].index, np.array(binned_correctness_mean.loc[i, :].values[:,0]), '-o', color=color)
        #print(binned_correctness_mean.loc[i, :].index)
        #print(binned_correctness_mean.loc[i, :].values[:,0])
        #print(np.array(binned_correctness_sem.loc[i, :].values[:,0]))

        pl.errorbar(binned_correctness_mean.loc[i, :].index,
                    binned_correctness_mean.loc[i, :].values[:,0],
                    np.array(binned_correctness_sem.loc[i, :].values[:,0]), color=color)

    pl.subplot(2, 2, 4)
    pl.ylim([45, 91])
    pl.plot(binned_same_direction_mean.index, binned_same_direction_mean.values, '-o', color=color)
    pl.errorbar(binned_same_direction_mean.index, binned_same_direction_mean.values, np.array(binned_same_direction_sem.values[:,0]), color=color)


pl.savefig(root_path / "behavior_data.png")
pl.show()



