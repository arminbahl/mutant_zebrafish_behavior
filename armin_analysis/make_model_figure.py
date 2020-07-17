import pylab as pl
from get_fish_info import get_fish_info
from fit_integrator_model import get_model_result, get_target_result
import numpy as np
from pathlib import Path
import gmm_model_fit
import pandas as pd
import pylab as pl
from fit_integrator_model import leaky_integrator_model2

experiment = "surrogate_fish2"
root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence")

for experiment in ["surrogate_fish2"]:

    estimated_parameters = dict({"repeat": [],
                                 "genotype": [],
                                 "tau": [],
                                 "noise_sigma": [],
                                 "T": [],
                                 "bout_clock_probability_below_threshold": [],
                                 "bout_clock_probability_above_threshold": [],
                                 })

    for repeat in range(12):

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

            X = np.load(root_path / experiment / f"leaky_integrator_model2_X_{genotype}_{repeat}.npy")
            F = np.load(root_path / experiment / f"leaky_integrator_model2_F_{genotype}_{repeat}.npy")

            # Create a single error function from the 5 error functions


            # Normalize the features based on their genera unit range
            for feature in range(5):
                norm_factor = np.percentile(F[0, :, feature], 25)

                for gen in range(F.shape[0]):
                    F[gen, :, feature] = F[gen, :, feature] / norm_factor # np.percentile(F[gen, :, feature], 75)

            F_sum = (F[:, :, 0] + F[:, :, 1] + F[:, :, 2] + F[:, :, 3] + F[:, :, 4]) / 5

            # gens = np.arange(0, F.shape[0]) + 1
            # pl.figure()
            # sc = pl.plot(gens, F[:, :, 0].min(axis=1), linestyle='--', marker='.', c='C0')
            # sc = pl.plot(gens, F[:, :, 1].min(axis=1), linestyle='--', marker='.', c='C1')
            # sc = pl.plot(gens, F[:, :, 2].min(axis=1), linestyle='--', marker='.', c='C2')
            # sc = pl.plot(gens, F[:, :, 3].min(axis=1), linestyle='--', marker='.', c='C3')
            # sc = pl.plot(gens, F[:, :, 4].min(axis=1), linestyle='--', marker='.', c='C4')
            # sc = pl.plot(gens, F_sum[:, :].min(axis=1), linestyle='--', marker='.', c='C5')
            #
            # pl.yscale("log")
            # pl.xlabel("Generation")
            # pl.ylabel("Log Error")
            # pl.show()

            # Get the best parameter
            best_i = np.argmin(F_sum[-1, :])
            tau, noise_sigma, T, bout_clock_probability_below_threshold, bout_clock_probability_above_threshold = X[-1, best_i]

            estimated_parameters["repeat"].append(repeat)
            estimated_parameters["genotype"].append(genotype)
            estimated_parameters["tau"].append(tau)
            estimated_parameters["noise_sigma"].append(noise_sigma)
            estimated_parameters["T"].append(T)
            estimated_parameters["bout_clock_probability_below_threshold"].append(bout_clock_probability_below_threshold)
            estimated_parameters["bout_clock_probability_above_threshold"].append(bout_clock_probability_above_threshold)

            # Run the simulation with these parameters
            print(tau, noise_sigma, T, bout_clock_probability_below_threshold, bout_clock_probability_above_threshold)

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

        df.to_hdf(root_path / experiment / f"all_data_best_model_repeat{repeat}.h5", key="all_bouts", complevel=4)

        # Extract more detailed behavioral features
        df_extracted_features, df_extracted_binned_features, \
        df_extracted_binned_features_same_direction, \
        df_extracted_binned_features_heading_angle_change_histograms, \
        df_extracted_binned_features_inter_bout_interval_histograms, \
        df_gmm_fitting_results = get_fish_info(df)

        df_extracted_features.to_hdf(root_path / experiment / f"all_data_best_model_repeat{repeat}.h5", key="extracted_features", complevel=4)
        df_extracted_binned_features.to_hdf(root_path / experiment / f"all_data_best_model_repeat{repeat}.h5", key="extracted_binned_features",
                                            complevel=4)
        df_extracted_binned_features_same_direction.to_hdf(root_path / experiment / f"all_data_best_model_repeat{repeat}.h5",
                                                           key="extracted_binned_features_same_direction", complevel=4)
        df_extracted_binned_features_heading_angle_change_histograms.to_hdf(root_path / experiment / f"all_data_best_model_repeat{repeat}.h5",
                                                                            key="extracted_binned_features_heading_angle_change_histograms",
                                                                            complevel=4)
        df_extracted_binned_features_inter_bout_interval_histograms.to_hdf(root_path / experiment / f"all_data_best_model_repeat{repeat}.h5",
                                                                           key="extracted_binned_features_inter_bout_interval_histograms",
                                                                           complevel=4)
        df_gmm_fitting_results.to_hdf(root_path / experiment / f"all_data_best_model_repeat{repeat}.h5", key="gmm_fitting_results", complevel=4)


    # Save the parameter info file
    df = pd.DataFrame.from_dict(estimated_parameters).astype(dtype={"repeat": "int64"}, copy=False)

    print(df)

    df.set_index(["genotype", 'repeat'], inplace=True)
    df.sort_index(inplace=True)

    df.to_excel(root_path / experiment / "estimated_model_parameters.xlsx", sheet_name="sheet1")