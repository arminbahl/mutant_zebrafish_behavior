import numpy as np
import pandas as pd
from gmm_model_fit import gmm_model_fit

def get_fish_info(df):
    fishes_IDs = df.index.get_level_values('fish_ID').unique().values

    df["distance_to_center"] = np.sqrt(df["bout_x"]**2 + df["bout_y"]**2)
    df["correct"] = df["heading_angle_change"].values > 0

    extracted_features = []
    extracted_binned_features = []
    extracted_binned_features_same_direction = []
    extracted_binned_features_heading_angle_change_histograms = []
    extracted_binned_features_inter_bout_interval_histograms = []

    for fish_ID in fishes_IDs:

        # Each fish only has a single genotype, so take the first value from the other index level
        genotype = df.loc[fish_ID, :, :, :].index.get_level_values(1).values[0]

        for stim_ID in range(4):

            extracted_features.append([fish_ID,
                                       genotype,
                                       stim_ID,
                                       df.loc[fish_ID, :, :, stim_ID].query("bout_time >= 10 and bout_time < 20 and distance_to_center < 0.95")["correct"].mean()*100,
                                       df.loc[fish_ID, :, :, stim_ID].query("bout_time >= 10 and bout_time < 20 and distance_to_center < 0.95")["inter_bout_interval"].mean()])

            for bin in [7, 9, 11, 13, 15, 17, 19, 21, 23]:
                values_in_bin = df.loc[fish_ID, :, :, stim_ID].query("bout_time >= @bin - 1 and bout_time < @bin + 1 and distance_to_center < 0.95")["correct"]

                extracted_binned_features.append([fish_ID,
                                                  genotype,
                                                  stim_ID,
                                                  bin,
                                                  values_in_bin.mean()*100 if len(values_in_bin) > 6 else np.nan])

            if stim_ID == 0:
                for bin in [0.125, 0.375, 0.625, 0.875, 1.125, 1.375]:
                    values_in_bin = df.loc[fish_ID, :, :, stim_ID].query("bout_time >= 5 and inter_bout_interval >= @bin - 0.125 and inter_bout_interval < @bin + 0.125 and distance_to_center < 0.95")["same_as_previous"]

                    extracted_binned_features_same_direction.append([fish_ID,
                                                                     genotype,
                                                                     bin,
                                                                     values_in_bin.mean()*100 if len(values_in_bin) > 6 else np.nan])

            # Histogram of angle change
            hist, bin_edges = np.histogram(df.loc[fish_ID, :, :, stim_ID].query("bout_time >= 10 and bout_time < 20 and distance_to_center < 0.95")["heading_angle_change"],
                                           bins=np.linspace(-120, 120, 40), density=False)
            hist = hist / hist.sum()  # Make it a probability histogram

            for i in range(len(hist)):
                extracted_binned_features_heading_angle_change_histograms.append([fish_ID, genotype, stim_ID,  (bin_edges[i] + bin_edges[i + 1]) / 2, hist[i]])

            # Histogram of inter-bout interval
            hist, bin_edges = np.histogram(df.loc[fish_ID, :, :, stim_ID].query("bout_time >= 10 and bout_time < 20 and distance_to_center < 0.95")["inter_bout_interval"],
                                           bins=np.linspace(0, 5, 40), density=False)
            hist = hist / hist.sum() # Make it a probability histogram

            for i in range(len(hist)):
                extracted_binned_features_inter_bout_interval_histograms.append([fish_ID, genotype, stim_ID, (bin_edges[i] + bin_edges[i + 1]) / 2, hist[i]])


    df_extracted_features = pd.DataFrame(extracted_features, columns=["fish_ID",
                                                                      "genotype",
                                                                      "stim",
                                                                      "correctness",
                                                                      "inter_bout_interval"])
    df_extracted_features.set_index(['fish_ID', "genotype", 'stim'], inplace=True)
    df_extracted_features.sort_index(inplace=True)

    ###############
    df_extracted_binned_features = pd.DataFrame(extracted_binned_features, columns=["fish_ID",
                                                                                    "genotype",
                                                                                    "stim",
                                                                                    "bin",
                                                                                    "correctness"])
    df_extracted_binned_features.set_index(['fish_ID', "genotype", 'stim', "bin"], inplace=True)
    df_extracted_binned_features.sort_index(inplace=True)

    ###############
    df_extracted_binned_features_same_direction = pd.DataFrame(extracted_binned_features_same_direction, columns=["fish_ID",
                                                                                                                  "genotype",
                                                                                                                  "bin",
                                                                                                                  "same_direction"])
    df_extracted_binned_features_same_direction.set_index(["fish_ID", "genotype", "bin"], inplace=True)
    df_extracted_binned_features_same_direction.sort_index(inplace=True)


    ###############
    df_extracted_binned_features_heading_angle_change_histograms = pd.DataFrame(extracted_binned_features_heading_angle_change_histograms,
                                                                            columns=["fish_ID",
                                                                                     "genotype",
                                                                                     "stim",
                                                                                     "bin",
                                                                                     "probability"])

    df_extracted_binned_features_heading_angle_change_histograms.set_index(["fish_ID", "genotype", "stim", "bin"], inplace=True)
    df_extracted_binned_features_heading_angle_change_histograms.sort_index(inplace=True)

    ###############
    df_extracted_binned_features_inter_bout_interval_histograms = pd.DataFrame(extracted_binned_features_inter_bout_interval_histograms,
                                                                                columns=["fish_ID",
                                                                                         "genotype",
                                                                                         "stim",
                                                                                         "bin",
                                                                                         "probability"])

    df_extracted_binned_features_inter_bout_interval_histograms.set_index(["fish_ID", "genotype", "stim", "bin"], inplace=True)
    df_extracted_binned_features_inter_bout_interval_histograms.sort_index(inplace=True)

    df = df_extracted_binned_features_heading_angle_change_histograms.groupby(["stim", "bin"]).mean()
    gmm_fitting_results = []
    for stim_ID in range(4):

        bins = df.loc[stim_ID, :]["probability"].index.values
        probabilities = df.loc[stim_ID, :]["probability"].values
        fit_w, fit_m, fit_s = gmm_model_fit(bins, probabilities)

        gmm_fitting_results.append([stim_ID] + list(fit_w) + list(fit_m) + list(fit_s))

    df_gmm_fitting_results = pd.DataFrame(gmm_fitting_results, columns=["stim", "w_left", "w_center", "w_right", "m_left", "m_center", "m_right", "s_left", "s_center", "s_right"])
    df_gmm_fitting_results.set_index(['stim'], inplace=True)
    df_gmm_fitting_results.sort_index(inplace=True)


    return df_extracted_features, \
           df_extracted_binned_features, \
           df_extracted_binned_features_same_direction, \
           df_extracted_binned_features_heading_angle_change_histograms, \
           df_extracted_binned_features_inter_bout_interval_histograms, \
           df_gmm_fitting_results
