from get_fish_info import get_fish_info
import pandas as pd
from pathlib import Path
import pylab as pl
import pickle
import numpy as np

root_path = Path("/n/home10/abahl/engert_storage_armin/ariel_paper/free_swimming_behavior_data/dot_motion_coherence")

for experiment in ["chrna2a",
                   "disc1_hetinx",
                   "scn1lab_NIBR_20200708",
                   "scn1lab_zirc_20200710"]:

    if experiment == "chrna2a":
        fish_data = pd.read_excel(root_path / experiment / "genotype.xlsx", header=None)
        fish_data.columns = ['fish_ID', "genotype"]

    if experiment == "disc1_hetinx":
        fish_data = pd.read_excel(root_path / experiment / "genotype.xlsx", header=None)
        fish_data.columns = ['fish_ID', "genotype"]

    if experiment == "scn1lab_NIBR_20200708":
        fish_data = pd.read_excel(root_path / experiment / "genotype.xlsx", header=0)
        fish_data.columns = ['fish_ID', "pre_genotype", "genotype"] # the post genotype is the correct one

    if experiment == "scn1lab_zirc_20200710":
        fish_data = pd.read_excel(root_path / experiment / "genotype.xlsx", header=0)
        fish_data.columns = ['fish_ID', "pre_genotype", "genotype"]

    print(fish_data)
    # if experiment == "scn1lab_NIBR_20200708":
    #     fish_data.columns = ['fish_ID', "genotype"]
    # if len(fish_data.columns) == 2:
    #     fish_data.columns = ['fish_ID', "genotype"]
    # else:
    #     fish_data.columns = ['fish_ID', "genotype", "take"]

    all_data = []

    numtrials = 30

    for i in range(len(fish_data)):

        fish_ID = fish_data.iloc[i]["fish_ID"]
        genotype = fish_data.iloc[i]["genotype"]
        # take = fish_data.loc[i]["take"]

        if genotype == "wt" or "+/+" in genotype:
            genotype = 'wt'
        elif genotype == "ht" or "+/-" in genotype:
            genotype = 'het'
        elif genotype == "hm" or "-/-" in genotype:
            genotype = 'hom'
        else:
            print(fish_ID, genotype, "unknown genotype. Skipping.")

        # if len(fish_data.columns) == 3:
        #     if fish_data.loc[i]["take"] == 0:
        #        print(fish_ID, genotype, "ignore fish (not good swimming?).")
        #        continue

        for trial in range(0, numtrials):
            print(experiment, fish_ID, genotype, trial)

            try:
                f = open(root_path / experiment / fish_ID / "raw_data" / f"trial{trial:03d}.dat", 'rb')
                data = pickle.load(f)
                f.close()
            except:
                break

            for stim in range(8):

                bout_times = data[f"bouts_start_stimulus_{stim:03d}"]["timestamp"]
                bout_xs = data[f"bouts_start_stimulus_{stim:03d}"]["fish_position_x"]
                bout_ys = data[f"bouts_start_stimulus_{stim:03d}"]["fish_position_y"]
                bout_start_fish_accumulated_orientation = data[f"bouts_start_stimulus_{stim:03d}"][
                    "fish_accumulated_orientation"]
                bout_end_fish_accumulated_orientation = data[f"bouts_end_stimulus_{stim:03d}"][
                    "fish_accumulated_orientation"]

                heading_angle_changes = bout_end_fish_accumulated_orientation - bout_start_fish_accumulated_orientation

                # Turn responses to left-ward motion the after way around
                if stim in [0, 1, 2, 3]:
                    heading_angle_changes = -heading_angle_changes

                for i in range(1, len(bout_times)):
                    all_data.append([fish_ID,
                                     genotype,
                                     trial,
                                     stim % 4,
                                     bout_times[i],
                                     bout_xs[i],
                                     bout_ys[i],
                                     bout_times[i] - bout_times[i - 1],
                                     heading_angle_changes[i],
                                     np.sign(heading_angle_changes[i]) == np.sign(heading_angle_changes[i - 1])])

    df = pd.DataFrame(all_data, columns=["fish_ID",
                                         "genotype",
                                         "trial",
                                         "stim",
                                         "bout_time",
                                         "bout_x",
                                         "bout_y",
                                         "inter_bout_interval",
                                         "heading_angle_change",
                                         "same_as_previous"]).astype(dtype={"trial": "int64",
                                                                            "stim": "int64",
                                                                            "same_as_previous": "bool"}, copy=False)

    df.set_index(['fish_ID', "genotype", 'trial', 'stim'], inplace=True)
    df.sort_index(inplace=True)

    df.to_hdf(root_path / experiment / "all_data.h5", key="all_bouts", complevel=9)

    # Extract behavioral features
    df_extracted_features, df_extracted_binned_features, \
    df_extracted_binned_features_same_direction, \
    df_extracted_binned_features_heading_angle_change_histograms, \
    df_extracted_binned_features_inter_bout_interval_histograms, \
    df_gmm_fitting_results = get_fish_info(df)

    df_extracted_features.to_hdf(root_path / experiment / "all_data.h5", key="extracted_features", complevel=9)
    df_extracted_binned_features.to_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features", complevel=9)
    df_extracted_binned_features_same_direction.to_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_same_direction", complevel=9)
    df_extracted_binned_features_heading_angle_change_histograms.to_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_heading_angle_change_histograms", complevel=9)
    df_extracted_binned_features_inter_bout_interval_histograms.to_hdf(root_path / experiment / "all_data.h5", key="extracted_binned_features_inter_bout_interval_histograms", complevel=9)
    df_gmm_fitting_results.to_hdf(root_path / experiment / "all_data.h5", key="gmm_fitting_results", complevel=9)
