import numpy as np
from os.path import dirname, join as pjoin
import scipy.io as sio
import pylab as pl
from pathlib import Path
from scipy.interpolate import interp1d
from numba import jit

@jit
def create_random_population(x_particles, y_particles, orientation_particles,
                             arena_width, arena_height, n):

    for i in range(n*10000):
        while True:
            x = (np.random.random() * 2 - 1) * max(arena_width, arena_height)
            y = (np.random.random() * 2 - 1) * max(arena_width, arena_height)

            if x**2/(arena_width/2)**2 + y**2/(arena_height/2)**2 <= 1:
                break

        x_particles[i] = x
        y_particles[i] = y
        orientation_particles[i] = np.random.random()*360


n = 5

def get_chance_levels(arena_width, arena_height, n):

    x_particles = np.empty(10000*n)
    y_particles = np.empty(10000*n)
    orientation_particles = np.empty(10000*n)

    create_random_population(x_particles, y_particles, orientation_particles, arena_width, arena_height, n)

    x_particles = x_particles.reshape((-1, n))
    y_particles = y_particles.reshape((-1, n))
    orientation_particles = orientation_particles.reshape((-1, n))

    # only select particules that are inside the ellipse

    # calculate chance distance
    ds = []
    for k in range(n):
        for l in range(n):
            if k == l:
                continue

            ds.append(np.sqrt((x_particles[:, k] - x_particles[:, l])**2 +
                              (y_particles[:, k] - y_particles[:, l])**2).mean())

    chance_distance = np.mean(ds)

    x = np.cos(np.radians(orientation_particles)).mean(axis=1)
    y = np.sin(np.radians(orientation_particles)).mean(axis=1)

    chance_polarization = np.sqrt(x ** 2 + y ** 2).mean()

    return chance_distance, chance_polarization



def get_features(arena_width, arena_height, orientation_particles, x_particles, y_particles, ts):

    group_chance_group_spacing, group_chance_polarization = get_chance_levels(arena_width, arena_height, n)

    # Calculate average distance between fish over time
    distances = []
    for k in range(n):
        for l in range(n):
            if k == l:
                continue

            distances.append(np.sqrt((x_particles[:, k] - x_particles[:, l]) ** 2 +
                                     (y_particles[:, k] - y_particles[:, l]) ** 2))

    group_spacing_rel_chance = np.log(np.nanmean(np.nanmean(distances, axis=0)) / group_chance_group_spacing)


    x = np.cos(np.radians(orientation_particles)).mean(axis=1)
    y = np.sin(np.radians(orientation_particles)).mean(axis=1)

    group_polarization_rel_chance = np.log(np.nanmean(np.sqrt(x ** 2 + y ** 2)) / group_chance_polarization)

    new_time = np.arange(0, 300, 1)

    f = interp1d(ts, x_particles, axis=0, bounds_error=False)
    x_particles_filtered_cm = f(new_time)

    f = interp1d(ts, y_particles, axis=0, bounds_error=False)
    y_particles_filtered_cm = f(new_time)

    speed = np.sqrt(np.diff(x_particles_filtered_cm, axis=0) ** 2 +
                    np.diff(y_particles_filtered_cm, axis=0) ** 2)

    group_speed = np.nanmean(np.nanmean(speed, axis=1))

    return group_spacing_rel_chance, group_polarization_rel_chance, group_speed


#root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/free_swimming_WT_by_age")
root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/model_results")
# root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/model_results")
# experiments = ["scn1lab_NIBR_20200708", "scn1lab_zirc_20200710", "disc1_hetinx"]

genotype = 'hom'

for age in [7, 14, 21]:

    # if age == 7:
    #     experiments = list((root_path / "free_swimming_7_dpf_1").glob("*_wtab"))
    #
    # if age == 14:
    #     experiments = list((root_path / "free_swimming_14_dpf_1").glob("*WT_AB")) + list((root_path / "free_swimming_14_dpf_2").glob("*_AB"))
    #
    # if age == 21:
    #     experiments = list((root_path / "free_swimming_21_dpf_1").glob("*AB")) + list((root_path / "free_swimming_21_dpf_2").glob("*_AB"))


    if age == 7:
        experiments = list((root_path / "disc1_hetinx_7dpf" / "wt").glob("*"))

    if age == 14:
        experiments = list((root_path / "disc1_hetinx_14dpf" / "wt").glob("*"))

    if age == 21:
        experiments = list((root_path / "disc1_hetinx_21dpf" / "wt").glob("*"))


    group_polarization_rel_chance_repeats = []
    group_spacing_rel_chance_repeats = []
    group_speed_repeats = []

    for experiment in experiments:

        header = sio.loadmat(experiment / "Header_new.mat")
        ratio_pix_cm = header["ratio_pix_cm"][0][0]
        #print(ratio_pix_cm)
        #continue

        simple_data = sio.loadmat(experiment / "simple_data.mat")
        borders = simple_data["borders"]# / ratio_pix_cm

        #print(simple_data["body_angle"].shape)
        #continue

        max_l = 60000 # 14800
        borders_center_x = np.mean(borders[:, 0])
        borders_center_y = np.mean(borders[:, 1])

        arena_width = (borders[:, 0].max() - borders[:, 0].min())
        arena_height = (borders[:, 1].max() - borders[:, 1].min())

        print(arena_height, arena_width)
        continue
        orientation_particles = simple_data["body_angle"].T[:max_l,:]
        x_particles = simple_data["x_f"].T[:max_l,:] / ratio_pix_cm
        y_particles = simple_data["y_f"].T[:max_l,:] / ratio_pix_cm

        ts = simple_data["T"][0, :max_l]

        if ts.shape[0] != max_l or \
                x_particles.shape[0] != max_l or \
                x_particles.shape[0] != max_l or \
                y_particles.shape[0] != max_l:
            print(ts.shape, orientation_particles.shape, x_particles.shape, y_particles.shape)
            continue

        group_spacing_rel_chance, group_polarization_rel_chance, group_speed = get_features(arena_width, arena_height,
                                                                         orientation_particles,
                                                                         x_particles,
                                                                         y_particles,
                                                                         ts)

        group_polarization_rel_chance_repeats.append(group_polarization_rel_chance)
        group_spacing_rel_chance_repeats.append(group_spacing_rel_chance)
        group_speed_repeats.append(group_speed)

    np.save(root_path / f"group_polarization_rel_chance_{genotype}_age{age}dpf.npy", group_polarization_rel_chance_repeats)
    np.save(root_path / f"group_spacing_rel_chance_{genotype}_age{age}dpf.npy", group_spacing_rel_chance_repeats)
    np.save(root_path / f"group_speed_{genotype}_age{age}dpf.npy", group_speed_repeats)

