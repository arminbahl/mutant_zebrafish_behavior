from get_fish_info import get_fish_info
import pandas as pd
from pathlib import Path
import numpy as np
from fit_integrator_model import leaky_integrator_model2

root_path = Path("/n/home10/abahl/engert_storage_armin/ariel_paper/free_swimming_behavior_data/dot_motion_coherence")

for experiment in ["surrogate_fish1", "surrogate_fish2", "surrogate_fish3"]:

    all_data_experiment = dict({"fish_ID": [],
                                "genotype": [],
                                "trial": [],
                                "stim": [],
                                "bout_time": [],
                                "bout_x": [],
                                "bout_y": [],
                                "inter_bout_interval": [],
                                "heading_angle_change": [],
                                "same_as_previous": []})

    for genotype in ["wt", "het", "hom"]:
        print("Creating data for", experiment, genotype)

        if experiment == "surrogate_fish1":
            if genotype == "wt":
                tau = 0.8
                noise_sigma = 10
                T = 1
                bout_clock_probability_below_threshold = 0.008
                bout_clock_probability_above_threshold = 0.04

            if genotype == "het":
                tau = 1.6 # doubling the time constant
                noise_sigma = 10
                T = 1
                bout_clock_probability_below_threshold = 0.008
                bout_clock_probability_above_threshold = 0.04

            if genotype == "hom":
                tau = 2.5  # even stronger time constant
                noise_sigma = 10
                T = 1
                bout_clock_probability_below_threshold = 0.008
                bout_clock_probability_above_threshold = 0.04


        if experiment == "surrogate_fish2":
            if genotype == "wt":
                tau = 0.8
                noise_sigma = 10
                T = 1
                bout_clock_probability_below_threshold = 0.008
                bout_clock_probability_above_threshold = 0.04

            if genotype == "het":
                tau = 0.8
                noise_sigma = 15    # more noisy
                T = 1
                bout_clock_probability_below_threshold = 0.008
                bout_clock_probability_above_threshold = 0.04

            if genotype == "hom":
                tau = 0.8
                noise_sigma = 20   # even noisier
                T = 1
                bout_clock_probability_below_threshold = 0.008
                bout_clock_probability_above_threshold = 0.04


        if experiment == "surrogate_fish3":
            if genotype == "wt":
                tau = 0.5
                noise_sigma = 15
                T = 1.5
                bout_clock_probability_below_threshold = 0.01
                bout_clock_probability_above_threshold = 0.03

            if genotype == "het":
                tau = 0.8
                noise_sigma = 20
                T = 1.2
                bout_clock_probability_below_threshold = 0.007
                bout_clock_probability_above_threshold = 0.035

            if genotype == "hom":
                tau = 1
                noise_sigma = 25
                T = 0.9
                bout_clock_probability_below_threshold = 0.006
                bout_clock_probability_above_threshold = 0.05

        # Initialize the buffers
        dt = 0.01
        ts = np.arange(0, 30, dt)
        xs = np.empty_like(ts)
        all_data = np.empty((10000000, 10))  # Max allow 10 million bouts per simulation run

        bout_counter = leaky_integrator_model2(dt, ts, xs, all_data,
                                               tau,
                                               noise_sigma,
                                               T,
                                               bout_clock_probability_below_threshold,
                                               bout_clock_probability_above_threshold)

        all_data_experiment["fish_ID"].extend([f"fish_{genotype}_{int(fish_ID)}" for fish_ID in all_data[:bout_counter - 1, 0]])
        all_data_experiment["genotype"].extend([genotype] * (bout_counter - 1))
        all_data_experiment["trial"].extend(all_data[:bout_counter - 1, 2])
        all_data_experiment["stim"].extend(all_data[:bout_counter - 1, 3])
        all_data_experiment["bout_time"].extend(all_data[:bout_counter - 1, 4])
        all_data_experiment["bout_x"].extend(all_data[:bout_counter - 1, 5])
        all_data_experiment["bout_y"].extend(all_data[:bout_counter - 1, 6])
        all_data_experiment["inter_bout_interval"].extend(all_data[:bout_counter - 1, 7])
        all_data_experiment["heading_angle_change"].extend(all_data[:bout_counter - 1, 8])
        all_data_experiment["same_as_previous"].extend(all_data[:bout_counter - 1, 9])

    df = pd.DataFrame.from_dict(all_data_experiment).astype(dtype={"trial": "int64",
                                                                  "stim": "int64",
                                                                  "same_as_previous": "bool"}, copy=False)

    print(df)

    df.set_index(['fish_ID', "genotype", 'trial', 'stim'], inplace=True)
    df.sort_index(inplace=True)

    (root_path / experiment).mkdir(exist_ok=True)

    df.to_hdf(root_path / experiment / "all_data.h5", key="all_bouts", complevel=9)

    # Extract more detailed behavioral features
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
