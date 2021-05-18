from pathlib import Path
import numpy as np
import autograd.numpy as anp
from numba import jit
import random
import pandas as pd
from get_fish_info import get_fish_info
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import argparse
from sklearn.metrics import mean_squared_log_error


# Initialize the buffers
dt = 0.01
ts = np.arange(0, 30, dt)
xs = np.empty_like(ts)
all_data = np.empty((10000000, 10))  # Max allow 10 million bouts per simulation run

@jit(nopython=True)
def leaky_integrator_model1(dt, ts, xs, all_data, tau, noise_sigma, T, bout_clock_probability_below_threshold, bout_clock_probability_above_threshold):

    bout_counter = 0

    for fish_ID in range(8):
        for trial in range(300):
            for stim in range(4):

                xs[0] = 0
                previous_bout_time = -1
                previous_heading_angle_change = -1

                for i in range(1, len(ts)):
                    if 10 <= ts[i] <= 20:
                        coherence = [0, 0.25, 0.5, 1][stim]
                    else:
                        coherence = 0

                    dx = random.gauss(coherence, noise_sigma) - xs[i - 1]

                    xs[i] = xs[i - 1] + dx * dt / tau

                    if previous_bout_time != -1 and ts[i] - previous_bout_time <= 0.2:
                        continue

                    heading_angle_change = 0

                    if xs[i] > T:
                        if random.random() < bout_clock_probability_above_threshold:
                            heading_angle_change = random.gauss(30, 1)
                    elif xs[i] < -T:
                        if random.random() < bout_clock_probability_above_threshold:
                            heading_angle_change = random.gauss(-30, 1)
                    elif random.random() < bout_clock_probability_below_threshold:
                        if xs[i] < 0:
                            heading_angle_change = random.gauss(-30, 1)
                        else:
                            heading_angle_change = random.gauss(30, 1)

                    if heading_angle_change != 0:

                        if previous_bout_time != -1:
                            all_data[bout_counter, 0] = fish_ID
                            all_data[bout_counter, 1] = 0   # Genotype
                            all_data[bout_counter, 2] = trial
                            all_data[bout_counter, 3] = stim
                            all_data[bout_counter, 4] = ts[i]
                            all_data[bout_counter, 5] = 0
                            all_data[bout_counter, 6] = 0
                            all_data[bout_counter, 7] = ts[i] - previous_bout_time
                            all_data[bout_counter, 8] = heading_angle_change
                            all_data[bout_counter, 9] = np.sign(heading_angle_change) == np.sign(previous_heading_angle_change)

                            bout_counter += 1

                            if bout_counter >= all_data.shape[0]:  # Array is full, must stop (this should not happen)
                                return bout_counter

                        previous_bout_time = ts[i]
                        previous_heading_angle_change = heading_angle_change

    return bout_counter


@jit(nopython=True)
def leaky_integrator_model2(dt, ts, xs, all_data, tau, noise_sigma, T, bout_clock_probability_below_threshold, bout_clock_probability_above_threshold):

    bout_counter = 0

    for fish_ID in range(8):
        for trial in range(300):
            for stim in range(4):

                xs[0] = 0
                previous_bout_time = -1
                previous_heading_angle_change = -1

                for i in range(1, len(ts)):

                    if 10 <= ts[i] <= 20:
                        coherence = [0, 0.25, 0.5, 1][stim]
                    else:
                        coherence = 0

                    dx = random.gauss(coherence, noise_sigma) - xs[i - 1]

                    xs[i] = xs[i - 1] + dx * dt / tau

                    if previous_bout_time != -1 and ts[i] - previous_bout_time <= 0.2:
                        continue

                    heading_angle_change = 0

                    if xs[i] > T:
                        if random.random() < bout_clock_probability_above_threshold:
                            heading_angle_change = random.gauss(22, 25)
                    elif xs[i] < -T:
                        if random.random() < bout_clock_probability_above_threshold:
                            heading_angle_change = random.gauss(-22, 25)
                    elif random.random() < bout_clock_probability_below_threshold:
                        heading_angle_change = random.gauss(0, 5)

                    if heading_angle_change != 0:

                        if previous_bout_time != -1:
                            all_data[bout_counter, 0] = fish_ID
                            all_data[bout_counter, 1] = 0
                            all_data[bout_counter, 2] = trial
                            all_data[bout_counter, 3] = stim
                            all_data[bout_counter, 4] = ts[i]
                            all_data[bout_counter, 5] = 0
                            all_data[bout_counter, 6] = 0
                            all_data[bout_counter, 7] = ts[i] - previous_bout_time
                            all_data[bout_counter, 8] = heading_angle_change
                            all_data[bout_counter, 9] = np.sign(heading_angle_change) == np.sign(previous_heading_angle_change)

                            bout_counter += 1

                            if bout_counter >= all_data.shape[0]:  # Array is full, must stop (this should not happen)
                                return bout_counter

                        previous_bout_time = ts[i]
                        previous_heading_angle_change = heading_angle_change

    return bout_counter


def get_target_result(hdf5_path, genotype):
    df_extracted_features = pd.read_hdf(hdf5_path, key="extracted_features")
    df_extracted_binned_features = pd.read_hdf(hdf5_path, key="extracted_binned_features")
    df_extracted_binned_features_same_direction = pd.read_hdf(hdf5_path, key="extracted_binned_features_same_direction")
    df_extracted_binned_features_heading_angle_change_histograms = pd.read_hdf(hdf5_path, key="extracted_binned_features_heading_angle_change_histograms")
    df_extracted_binned_features_inter_bout_interval_histograms = pd.read_hdf(hdf5_path, key="extracted_binned_features_inter_bout_interval_histograms")
    df_gmm_fitting_results = pd.read_hdf(hdf5_path, key="gmm_fitting_results")

    return df_extracted_features.query("genotype == @genotype").groupby("stim").mean()["correctness"], \
           df_extracted_features.query("genotype == @genotype").groupby("stim").mean()["inter_bout_interval"], \
           df_extracted_binned_features.query("genotype == @genotype").groupby(["stim", "bin"]).mean()["correctness"], \
           df_extracted_binned_features_same_direction.query("genotype == @genotype").groupby(["bin"]).mean()["same_direction"], \
           df_extracted_binned_features_heading_angle_change_histograms.query("genotype == @genotype").groupby(["stim", "bin"]).mean()["probability"], \
           df_extracted_binned_features_inter_bout_interval_histograms.query("genotype == @genotype").groupby(["stim", "bin"]).mean()["probability"], \
           df_gmm_fitting_results


def get_model_result(x):

    tau = x[0]
    noise_sigma = x[1]
    T = x[2]
    bout_clock_probability_below_threshold = x[3]
    bout_clock_probability_above_threshold = x[4]

    bout_counter = leaky_integrator_model2(dt, ts, xs, all_data,
                                           tau,
                                           noise_sigma,
                                           T,
                                           bout_clock_probability_below_threshold,
                                           bout_clock_probability_above_threshold)

    df = pd.DataFrame(all_data[:bout_counter - 1],
                      columns=["fish_ID",
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

    df_extracted_features, \
    df_extracted_binned_features, \
    df_extracted_binned_features_same_direction, \
    df_extracted_binned_features_heading_angle_change_histograms, \
    df_extracted_binned_features_inter_bout_interval_histograms, \
    df_gmm_fitting_results = get_fish_info(df)

    return df_extracted_features.groupby("stim").mean()["correctness"], \
           df_extracted_features.groupby("stim").mean()["inter_bout_interval"], \
           df_extracted_binned_features.groupby(["stim", "bin"]).mean()["correctness"], \
           df_extracted_binned_features_same_direction.groupby(["bin"]).mean()["same_direction"], \
           df_extracted_binned_features_heading_angle_change_histograms.groupby(["stim", "bin"]).mean()["probability"], \
           df_extracted_binned_features_inter_bout_interval_histograms.groupby(["stim", "bin"]).mean()["probability"], \
           df_gmm_fitting_results


class MyProblem(Problem):

    def __init__(self, root_path, target_genotype,
                 **kwargs):

        super().__init__(n_var=5,
                         n_obj=5,
                         n_constr=0,
                         xl=anp.array([  0.1,   0,  0.1, 0.001, 0.001]),
                         xu=anp.array([ 15,   100,  5,   0.05,  0.05]),
                         elementwise_evaluation=True,
                         **kwargs)

        self.target_df_correctness_as_function_of_coherence, \
        self.target_df_inter_bout_interval_as_function_of_coherence, \
        self.target_df_binned_correctness, \
        self.target_df_binned_same_direction, \
        self.target_df_binned_features_heading_angle_change_histograms, \
        self.target_df_binned_features_inter_bout_interval_histograms, \
        self.target_df_gmm_fitting_results = get_target_result(root_path / "all_data.h5", target_genotype)

    #def compute_error(self, vals, target):
    #    return np.mean(((vals - target) / target)**2)
    def MSLE(self, y_true, y_pred):
        """Same as sklearn.metrics import mean_squared_log_error but should work with nans.
            Note that it cannot deal with negative values."""

        n = len(y_true)
        return np.nanmean([(np.log(y_pred[i] + 1) -
                            np.log(y_true[i] + 1)) ** 2.0 for i in range(n)])

    def _evaluate(self, x, out, *args, **kwargs):

        model_df_correctness_as_function_of_coherence, \
        model_df_inter_bout_interval_as_function_of_coherence, \
        model_df_binned_correctness, \
        model_df_binned_same_direction, \
        model_df_binned_features_heading_angle_change_histograms, \
        model_df_binned_features_inter_bout_interval_histograms, \
        model_df_gmm_fitting_results = get_model_result(x)

        # # Calculate the errors
        # e0 = ((model_df_correctness_as_function_of_coherence - self.target_df_correctness_as_function_of_coherence) ** 2).sum()
        # e1 = ((model_df_inter_bout_interval_as_function_of_coherence - self.target_df_inter_bout_interval_as_function_of_coherence) ** 2).sum()
        # e2 = ((model_df_binned_correctness.loc[1] - self.target_df_binned_correctness.loc[1]) ** 2).sum() + \
        #      ((model_df_binned_correctness.loc[2] - self.target_df_binned_correctness.loc[2]) ** 2).sum() + \
        #      ((model_df_binned_correctness.loc[3] - self.target_df_binned_correctness.loc[3]) ** 2).sum()
        # e3 = ((model_df_binned_same_direction - self.target_df_binned_same_direction) ** 2).sum()

        # Compute Error between estimated weights for the histograms
        #e4 = ((model_df_gmm_fitting_results["w_left"] - self.target_df_gmm_fitting_results["w_left"]) ** 2).sum() + \
        #     ((model_df_gmm_fitting_results["w_center"] - self.target_df_gmm_fitting_results["w_center"]) ** 2).sum() + \
        #     ((model_df_gmm_fitting_results["w_right"] - self.target_df_gmm_fitting_results["w_right"]) ** 2).sum()

        # 18. Mai 2021
        # Reviewer comment: Compute the errors in a different way, using the mean squared log error
        e0 = self.MSLE(model_df_correctness_as_function_of_coherence,
                                    self.target_df_correctness_as_function_of_coherence)
        e1 = self.MSLE(model_df_inter_bout_interval_as_function_of_coherence,
                                    self.target_df_inter_bout_interval_as_function_of_coherence)
        e2 = self.MSLE(model_df_binned_correctness.loc[1], self.target_df_binned_correctness.loc[1]) + \
             self.MSLE(model_df_binned_correctness.loc[2], self.target_df_binned_correctness.loc[2]) + \
             self.MSLE(model_df_binned_correctness.loc[3], self.target_df_binned_correctness.loc[3])
        e3 = self.MSLE(model_df_binned_same_direction, self.target_df_binned_same_direction)

        # Keep squared distance here
        e4 = ((model_df_gmm_fitting_results["w_left"] - self.target_df_gmm_fitting_results["w_left"]) ** 2).sum() + \
             ((model_df_gmm_fitting_results["w_center"] - self.target_df_gmm_fitting_results["w_center"]) ** 2).sum() + \
             ((model_df_gmm_fitting_results["w_right"] - self.target_df_gmm_fitting_results["w_right"]) ** 2).sum()

        #Save all error values
        #out["F"] = [e0+e1+e2+e3+e4+e5+e6]
        out["F"] = [e0, e1, e2, e3, e4]#, e2, e3, e4, e5, e6]

        # e0 = self.compute_error(model_df_correctness_as_function_of_coherence, self.target_df_correctness_as_function_of_coherence)
        # e1 = self.compute_error(model_df_inter_bout_interval_as_function_of_coherence, self.target_df_inter_bout_interval_as_function_of_coherence)
        # e2 = self.compute_error(model_df_binned_correctness.loc[1], self.target_df_binned_correctness.loc[1])*0.333 + \
        #      self.compute_error(model_df_binned_correctness.loc[2], self.target_df_binned_correctness.loc[2])*0.333 + \
        #      self.compute_error(model_df_binned_correctness.loc[3], self.target_df_binned_correctness.loc[3])*0.333
        # e3 = self.compute_error(model_df_binned_same_direction, self.target_df_binned_same_direction)
        # e4 = self.compute_error(model_df_gmm_fitting_results["w_left"], self.target_df_gmm_fitting_results["w_left"])*0.333 + \
        #      self.compute_error(model_df_gmm_fitting_results["w_center"], self.target_df_gmm_fitting_results["w_center"])*0.333 + \
        #      self.compute_error(model_df_gmm_fitting_results["w_right"], self.target_df_gmm_fitting_results["w_right"])*0.333
        #
        # out["F"] = [e0 + e1 + e2 + e3 + e4]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fits a behavioral model to experimental data.')

    #parser.add_argument('-rp', '--root_path', type=str, help='Path to experimental folder containing the all_data.h5 file.', required=True)
    parser.add_argument('-tg', '--target_genotype', type=str, help='The target genotype within the experimental data.', required=True)
    parser.add_argument('-exp', '--experiment_i', type=int, help='Experiment index.', required=True)
    parser.add_argument('-rs', '--random_seed', type=int, help='Random seed of the optimization algorithm.', required=True)

    args = parser.parse_args()

    experiment_i = args.experiment_i
    target_genotype = args.target_genotype
    random_seed = args.random_seed

    experiment_names = ["surrogate_fish1",
                        "surrogate_fish2",
                        "surrogate_fish3",
                        "scn1lab_sa16474", # measured before corona lock down
                        "scn1lab_NIBR",  # measured before corona lock down
                        "scn1lab_NIBR_20200708",  # measured after corona lock down
                        "scn1lab_zirc_20200710",  # measured after corona lock down
                        "immp2l_summer",  # membrane transporter in the mitochondirum
                        "immp2l_NIBR",
                        "disc1_hetinx",
                        "chrna2a"]  # not so important

    root_path = Path("/n/home10/abahl/engert_storage_armin/ariel_paper/free_swimming_behavior_data/dot_motion_coherence") / experiment_names[experiment_i]

    print(f"Starting. target_genotype: {target_genotype}; optimization_repeat: {random_seed}, root_path: {root_path}")

    problem = MyProblem(root_path, target_genotype, parallelization=("threads", 51))

    algorithm = NSGA2(
        pop_size=51*8*2,
        n_offsprings=51*8,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True)

    termination = get_termination("n_gen", 80)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=random_seed,
                   pf=problem.pareto_front(use_cache=False),
                   save_history=True,
                   verbose=True)

    # Collect the population in each generation
    pop_each_gen = [a.pop for a in res.history]
    F_each_gen = [pop.get("F") for pop in pop_each_gen]
    X_each_gen = [pop.get("X") for pop in pop_each_gen]

    print("Done...")
    print("obj_each_gen", np.array(F_each_gen).shape)
    print("X_each_gen", np.array(X_each_gen).shape)

    # Save optimized values
    np.save(root_path / f"review1_leaky_integrator_model2_X_{target_genotype}_{random_seed}.npy", np.array(X_each_gen))
    np.save(root_path / f"review1_leaky_integrator_model2_F_{target_genotype}_{random_seed}.npy", np.array(F_each_gen))
