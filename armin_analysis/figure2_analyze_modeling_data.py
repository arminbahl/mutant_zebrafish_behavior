import pylab as pl
from get_fish_info import get_fish_info
from fit_integrator_model import get_model_result, get_target_result
import numpy as np
from pathlib import Path
import gmm_model_fit
import pandas as pd
import pylab as pl
from fit_integrator_model import leaky_integrator_model2

#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence")
root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/mutant_behavior_data/dot_motion_coherence")

review_string = "review1_"

#for experiment in ["disc1_hetinx", "scn1lab_zirc_20200710", "scn1lab_NIBR_20200708"]:#["scn1lab_sa16474"]:#["surrogate_fish1", "surrogate_fish2", "surrogate_fish3"]:
for experiment in ["surrogate_fish1"]:#, "surrogate_fish3"]:

    if experiment == "disc1_hetinx":
        genotypes = ["wt", "het", "hom"]
    elif experiment == "scn1lab_NIBR_20200708" or experiment == "scn1lab_zirc_20200710":
        genotypes = ["wt", "het"]
    else:
        genotypes = ["wt"]

    estimated_parameters = dict({"repeat": [],
                                 "genotype": [],
                                 "tau": [],
                                 "noise_sigma": [],
                                 "T": [],
                                 "bout_clock_probability_below_threshold": [],
                                 "bout_clock_probability_above_threshold": [],
                                 })


    errors_over_generations = dict({"repeat": [],
                                  "genotype": [],
                                  "generation": [],
                                  "error_i": [],
                                  "error": []})
    for repeat in [2,3,4,9,10]:#range(12):

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


        for genotype in genotypes:

            X = np.load(root_path / experiment / f"{review_string}leaky_integrator_model2_X_{genotype}_{repeat}.npy")
            F = np.load(root_path / experiment / f"{review_string}leaky_integrator_model2_F_{genotype}_{repeat}.npy")

            # Create a single error function from the 5 error functions
            # Normalize the features based on their genera unit range
            for error_i in range(5):
                norm_factor = np.percentile(F[0, :, error_i], 25)

                for generation in range(F.shape[0]):
                    F[generation, :, error_i] = F[generation, :, error_i] / norm_factor # np.percentile(F[gen, :, feature], 75)

            F_sum = (F[:, :, 0] + F[:, :, 1] + 5*F[:, :, 2] + F[:, :, 3] + F[:, :, 4]) / 5

            for generation in range(F.shape[0]):
                    for error_i in range(6):
                        errors_over_generations["repeat"].append(repeat)
                        errors_over_generations["genotype"].append(genotype)
                        errors_over_generations["generation"].append(generation)
                        errors_over_generations["error_i"].append(error_i)
                        if error_i < 5:
                            errors_over_generations["error"].append(F[generation, :, error_i].min())
                        else:
                            errors_over_generations["error"].append(F_sum[generation, :].min())

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


        df.set_index(['fish_ID', "genotype", 'trial', 'stim'], inplace=True)
        df.sort_index(inplace=True)

        (root_path / experiment).mkdir(exist_ok=True)

        df.to_hdf(root_path / experiment / f"{review_string}all_data_best_model_repeat{repeat}.h5", key="all_bouts", complevel=4)

        # Extract more detailed behavioral features
        df_extracted_features, df_extracted_binned_features, \
        df_extracted_binned_features_same_direction, \
        df_extracted_binned_features_heading_angle_change_histograms, \
        df_extracted_binned_features_inter_bout_interval_histograms, \
        df_gmm_fitting_results = get_fish_info(df)

        df_extracted_features.to_hdf(root_path / experiment / f"{review_string}all_data_best_model_repeat{repeat}.h5", key="extracted_features", complevel=4)
        df_extracted_binned_features.to_hdf(root_path / experiment / f"{review_string}all_data_best_model_repeat{repeat}.h5", key="extracted_binned_features",
                                            complevel=4)
        df_extracted_binned_features_same_direction.to_hdf(root_path / experiment / f"{review_string}all_data_best_model_repeat{repeat}.h5",
                                                           key="extracted_binned_features_same_direction", complevel=4)
        df_extracted_binned_features_heading_angle_change_histograms.to_hdf(root_path / experiment / f"{review_string}all_data_best_model_repeat{repeat}.h5",
                                                                            key="extracted_binned_features_heading_angle_change_histograms",
                                                                            complevel=4)
        df_extracted_binned_features_inter_bout_interval_histograms.to_hdf(root_path / experiment / f"{review_string}all_data_best_model_repeat{repeat}.h5",
                                                                           key="extracted_binned_features_inter_bout_interval_histograms",
                                                                           complevel=4)
        df_gmm_fitting_results.to_hdf(root_path / experiment / f"{review_string}all_data_best_model_repeat{repeat}.h5", key="gmm_fitting_results", complevel=4)


    # Save the parameter info file
    df = pd.DataFrame.from_dict(estimated_parameters).astype(dtype={"repeat": "int64"}, copy=False)
    df.set_index(["genotype", 'repeat'], inplace=True)
    df.sort_index(inplace=True)

    df.to_excel(root_path / experiment / f"{review_string}estimated_model_parameters.xlsx", sheet_name="data")
    df.to_hdf(root_path / experiment / f"{review_string}estimated_model_parameters.h5", key="data")


    """
    #######
    # Also run the consensus model simulation
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
        df_estimated_parameters = df.query("genotype == @genotype").droplevel(["genotype"])

        # Initialize the buffers
        dt = 0.01
        ts = np.arange(0, 30, dt)
        xs = np.empty_like(ts)
        all_data = np.empty((10000000, 10))  # Max allow 10 million bouts per simulation run

        tau = np.median(df_estimated_parameters['tau'].values)
        noise_sigma = np.median(df_estimated_parameters['noise_sigma'])
        T = np.median(df_estimated_parameters['T'])
        bout_clock_probability_below_threshold = np.median(df_estimated_parameters['bout_clock_probability_below_threshold'])
        bout_clock_probability_above_threshold = np.median(df_estimated_parameters['bout_clock_probability_above_threshold'])

        bout_counter = leaky_integrator_model2(dt, ts, xs, all_data,
                                               tau,
                                               noise_sigma,
                                               T,
                                               bout_clock_probability_below_threshold,
                                               bout_clock_probability_above_threshold)

        print(bout_counter)

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

    df.to_hdf(root_path / experiment / f"all_data_best_model_consensus.h5", key="all_bouts", complevel=4)


    # Extract more detailed behavioral features
    df_extracted_features, df_extracted_binned_features, \
    df_extracted_binned_features_same_direction, \
    df_extracted_binned_features_heading_angle_change_histograms, \
    df_extracted_binned_features_inter_bout_interval_histograms, \
    df_gmm_fitting_results = get_fish_info(df)

    df_extracted_features.to_hdf(root_path / experiment / "all_data_best_model_consensus.h5", key="extracted_features", complevel=9)
    df_extracted_binned_features.to_hdf(root_path / experiment / "all_data_best_model_consensus.h5", key="extracted_binned_features", complevel=9)
    df_extracted_binned_features_same_direction.to_hdf(root_path / experiment / "all_data_best_model_consensus.h5", key="extracted_binned_features_same_direction", complevel=9)
    df_extracted_binned_features_heading_angle_change_histograms.to_hdf(root_path / experiment / "all_data_best_model_consensus.h5", key="extracted_binned_features_heading_angle_change_histograms", complevel=9)
    df_extracted_binned_features_inter_bout_interval_histograms.to_hdf(root_path / experiment / "all_data_best_model_consensus.h5", key="extracted_binned_features_inter_bout_interval_histograms", complevel=9)
    df_gmm_fitting_results.to_hdf(root_path / experiment / "all_data_best_model_consensus.h5", key="gmm_fitting_results", complevel=9)
    """



    df = pd.DataFrame.from_dict(errors_over_generations).astype(dtype={"repeat": "int64",
                                                                       "generation":"int64",
                                                                       "error_i":"int64"}, copy=False)
    df.set_index(["genotype", 'repeat', "error_i", "generation"], inplace=True)
    df.sort_index(inplace=True)

    df.to_excel(root_path / experiment / f"{review_string}errors_over_generations.xlsx", sheet_name="data")
    df.to_hdf(root_path / experiment / f"{review_string}errors_over_generations.h5", key="data")


