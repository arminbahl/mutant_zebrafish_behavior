from numba import jit
import numpy as np
import random
import pylab as pl
import imageio
from pathlib import Path
import cv2
from tqdm import tqdm
from PIL import Image
from matplotlib.colors import to_rgb
import pandas as pd
from scipy.interpolate import interp1d
import scipy.io as sio

tau_forward_speed = 0.1  # s
tau_turning_speed = 0.1  # s
fish_size = 20  # 20 pixel (one body length = 2 mm);
arena_size = 1024  # arena is 1024 pixel ( = 102 mm)
fish_forward_speed = 20  # 20 pixels per bout
dt = 0.01
n = 5


@jit(nopython=True)
def determine_angle_between_fish(ang1, x1, y1, x2, y2):
    ang2 = np.degrees(np.arctan2(y2-y1, x2-x1))
    if ang2 < 0:
        ang2 = 360 + ang2

    if ang2 > ang1:
        ang2 -= 360

    angle_between_fish = (ang1 - ang2)

    return angle_between_fish

@jit(nopython=True)
def simulate_particles(tau,
                       sigma,
                       T,
                       p_below,
                       p_above,
                       effect_strength_motion,
                       effect_strength_clutter,
                       dt,
                       ts,
                       integrated_motion_perception_particles,
                       current_clutter_perception_particles,
                       x_particles,
                       y_particles,
                       orientation_particles,
                       turning_speed_particles,
                       forward_speed_particles):

    n_particles = integrated_motion_perception_particles.shape[1]

    for time_i in range(1, ts.shape[0]):
        ts[time_i] = ts[time_i - 1] + dt

        for particle_i in range(n_particles):

            current_motion_perception_particle_i = 0
            current_clutter_perception_particle_i = 0

            # Perception is always on
            for particle_j in range(n_particles):
                if particle_i == particle_j:
                    continue

                distance = np.sqrt(
                    (x_particles[time_i - 1, particle_i] - x_particles[time_i - 1, particle_j]) ** 2 +
                    (y_particles[time_i - 1, particle_i] - y_particles[time_i - 1, particle_j]) ** 2)

                ang1 = orientation_particles[time_i - 1, particle_i]
                ang2 = orientation_particles[time_i - 1, particle_j]

                if ang2 > ang1:
                    angle_between_orientation_vectors = ang1 - ang2 + 360
                else:
                    angle_between_orientation_vectors = ang1 - ang2

                # is it on the right or the left side?
                angle_between_fish = determine_angle_between_fish(ang1,
                                                x_particles[time_i - 1, particle_i],
                                                y_particles[time_i - 1, particle_i],
                                                x_particles[time_i - 1, particle_j],
                                                y_particles[time_i - 1, particle_j])

                perpendicular_motion_drive = effect_strength_motion * \
                                             forward_speed_particles[time_i - 1, particle_j] * \
                                             np.sin(np.radians(angle_between_orientation_vectors)) / distance

                # Generally, fish are repulsive based on angle the other fish casts on the retina!
                size_on_retina = 2*np.arctan(fish_size/(2*distance))

                if 0 <= angle_between_fish < 180:
                    current_clutter_perception_particle_i += effect_strength_clutter * size_on_retina
                if 180 <= angle_between_fish < 360:
                    current_clutter_perception_particle_i -= effect_strength_clutter * size_on_retina

                # outward motion
                if 0 <= angle_between_fish < 180 and perpendicular_motion_drive > 0:
                    current_motion_perception_particle_i += perpendicular_motion_drive#*0.7

                # inward motion
                if 0 <= angle_between_fish < 180 and perpendicular_motion_drive <= 0:
                    current_motion_perception_particle_i += perpendicular_motion_drive#*1.4  # want to keep it on average the same as before

                # inward motion
                if 180 <= angle_between_fish < 360 and perpendicular_motion_drive > 0:
                    current_motion_perception_particle_i += perpendicular_motion_drive#*1.4

                # outward motion
                if 180 <= angle_between_fish < 360 and perpendicular_motion_drive <= 0:
                    current_motion_perception_particle_i += perpendicular_motion_drive#*0.7

            # # motion to the right
            # if 100 <= ts[time_i] < 160:
            #     current_motion_perception_particle_i += np.sin(np.radians(orientation_particles[time_i - 1, particle_i]))
            # # motion to the left
            # if 180 <= ts[time_i] < 240:
            #     current_motion_perception_particle_i += -np.sin(np.radians(orientation_particles[time_i - 1, particle_i]))

            # temporally integrate motion cues
            dk = random.gauss(current_motion_perception_particle_i, sigma) - integrated_motion_perception_particles[time_i - 1, particle_i]
            integrated_motion_perception_particles[time_i, particle_i] = integrated_motion_perception_particles[time_i - 1, particle_i] + dk * dt / tau

            # Save the momentary clutter percept
            current_clutter_perception_particles[time_i, particle_i] = current_clutter_perception_particle_i

            turning_speed_particle_i = 0 / dt
            forward_speed_particle_i = 0 / dt

            current_perception_particle_i = integrated_motion_perception_particles[time_i, particle_i] + \
                                            current_clutter_perception_particles[time_i, particle_i]

            if current_perception_particle_i > T:
                if random.random() < p_above:
                    turning_speed_particle_i = random.gauss(-22, 25) / dt  # negative angles are right turns
                    forward_speed_particle_i = fish_forward_speed / dt
            elif current_perception_particle_i < -T:
                if random.random() < p_above:
                    turning_speed_particle_i = random.gauss(22, 25) / dt  # positive angles are left turns
                    forward_speed_particle_i = fish_forward_speed / dt
            else:
                if random.random() < p_below:
                    turning_speed_particle_i = random.gauss(0, 5) / dt
                    forward_speed_particle_i = fish_forward_speed / dt

            dk = turning_speed_particle_i - turning_speed_particles[time_i - 1, particle_i]
            turning_speed_particles[time_i, particle_i] = turning_speed_particles[time_i - 1, particle_i] + dk * dt / tau_turning_speed

            dk = forward_speed_particle_i - forward_speed_particles[time_i - 1, particle_i]
            forward_speed_particles[time_i, particle_i] = forward_speed_particles[time_i - 1, particle_i] + dk * dt / tau_forward_speed

            orientation_particles[time_i, particle_i] = orientation_particles[time_i - 1, particle_i] + \
                                                        turning_speed_particles[time_i, particle_i] * dt

            x_particles[time_i, particle_i] = x_particles[time_i - 1, particle_i] + \
                                              forward_speed_particles[time_i, particle_i] * dt * np.cos(np.radians(orientation_particles[time_i, particle_i]))
            y_particles[time_i, particle_i] = y_particles[time_i - 1, particle_i] + \
                                              forward_speed_particles[time_i, particle_i] * dt * np.sin(np.radians(orientation_particles[time_i, particle_i]))

            recompute_orientation = False
            #x_particles[time_i, particle_i] = x_particles[time_i, particle_i] % 512
            #y_particles[time_i, particle_i] = y_particles[time_i, particle_i] % 512

            #bounce of the circlular wall
            if ((x_particles[time_i, particle_i] - arena_size/2)**2 +
                (y_particles[time_i, particle_i]-arena_size/2)**2) >= (arena_size/2)**2:

                x_particles[time_i, particle_i] = x_particles[time_i - 1, particle_i]
                y_particles[time_i, particle_i] = y_particles[time_i - 1, particle_i]

                orientation_particles[time_i, particle_i] = np.random.random()*360

            #     recompute_orientation = True
            #
            # if y_particles[time_i, particle_i] >= arena_size:
            #     y_particles[time_i, particle_i] = x_particles[time_i - 1, particle_i]
            #     recompute_orientation = True
            #
            # if x_particles[time_i, particle_i] < 0:
            #     x_particles[time_i, particle_i] = 0 - (x_particles[time_i, particle_i] - 0)
            #     recompute_orientation = True
            #
            # if y_particles[time_i, particle_i] < 0:
            #     y_particles[time_i, particle_i] = 0 - (y_particles[time_i, particle_i] - 0)
            #     recompute_orientation = True
            #
            # if recompute_orientation == True:
            #     orientation_particles[time_i, particle_i] = np.degrees(np.arctan2(y_particles[time_i, particle_i] - y_particles[time_i - 1, particle_i],
            #                                                                       x_particles[time_i, particle_i] - x_particles[time_i - 1, particle_i]))

            orientation_particles[time_i, particle_i] = orientation_particles[time_i, particle_i] % 360



# Load the consensus parameter sets
root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/dot_motion_coherence")
root_output_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/model_results")

for age in [7, 14, 21]:
    for experiment in ["scn1lab_NIBR_20200708", "scn1lab_zirc_20200710", "disc1_hetinx"]:

        if experiment == "scn1lab_NIBR_20200708" or experiment == "scn1lab_zirc_20200710":
            genotypes = ["wt", "het"]

        if experiment == 'disc1_hetinx':
            genotypes = ["wt", "hom"]

        output_path = root_output_path / f"{experiment}_{age}dpf"
        output_path.mkdir(exist_ok=True)

        for genotype in genotypes:

            df_estimated_parameters_model = pd.read_hdf(root_path / experiment / "estimated_model_parameters.h5", key="data").query("genotype == @genotype").droplevel(["genotype"])

            polarizations_repeats = []
            ds_over_time_repeats = []
            speed_over_time_repeats = []


            for j in range(12):

                tau = df_estimated_parameters_model['tau'][j]
                sigma = df_estimated_parameters_model['noise_sigma'][j]
                T = df_estimated_parameters_model['T'][j]
                p_below = df_estimated_parameters_model['bout_clock_probability_below_threshold'][j]
                p_above = df_estimated_parameters_model['bout_clock_probability_above_threshold'][j]

                polarizations = []
                ds_over_time = []
                speed_over_time = []

                ts = np.zeros(int(600/dt))
                integrated_motion_perception_particles = np.zeros((int(600/dt), n))
                current_clutter_perception_particles = np.zeros((int(600/dt), n))

                x_particles = np.zeros((int(600/dt), n))
                y_particles = np.zeros((int(600/dt), n))
                orientation_particles = np.zeros((int(600/dt), n)) * 360
                forward_speed_particles = np.zeros((int(600/dt), n))
                turning_speed_particles = np.zeros((int(600/dt), n))

                for i in range(n):
                    while True:
                        x = np.random.random() * arena_size
                        y = np.random.random() * arena_size

                        if (x - arena_size/2)**2 + (y - arena_size/2)**2 < (arena_size/2)**2:
                            break

                    x_particles[0, i] = x
                    y_particles[0, i] = y
                    orientation_particles[0, i] = np.random.random()*360


                effect_strength_motion = 1

                if age == 7:
                    effect_strength_clutter = -3  # repulsive
                if age == 14:
                    effect_strength_clutter = 0  # neutral
                if age == 21:
                    effect_strength_clutter = 3  # attractive

                #effect_strength_motion = 0
                #effect_strength_clutter = 0

                if experiment == "scn1lab_NIBR_20200708" and genotype == "wt":
                    pass

                if experiment == "scn1lab_NIBR_20200708" and genotype == "het":
                    effect_strength_clutter -= 2 # make it more repulsive

                if experiment == "scn1lab_zirc_20200710" and genotype == "wt":
                    pass

                if experiment == "scn1lab_zirc_20200710" and genotype == "het":
                    effect_strength_clutter -= 2 # make it more repulsive

                if experiment == "disc1_hetinx" and genotype == "wt":
                    pass

                if experiment == "disc1_hetinx" and genotype == "hom":
                    effect_strength_clutter += 2 # make it more attractive


                simulate_particles(tau, sigma, T, p_below, p_above,
                                   effect_strength_motion,
                                   effect_strength_clutter,
                                   dt,
                                   ts,
                                   integrated_motion_perception_particles,
                                   current_clutter_perception_particles,
                                   x_particles,
                                   y_particles,
                                   orientation_particles,
                                   turning_speed_particles,
                                   forward_speed_particles)

                if j == 0:
                    writer = imageio.get_writer(output_path / f'collective_model_example_{genotype}.mp4', codec='libx264', fps=30,
                                                ffmpeg_params=["-b:v", "8M"])

                    node_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

                    for time_i in tqdm(np.arange(0, ts.shape[0] - 1, 20)):

                        img = np.zeros((arena_size, arena_size), dtype=np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                        cv2.putText(img, f"{ts[time_i]:.1f} s", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
                                    color=(255, 255, 255))

                        # if 20 < ts[time_i] < 100:
                        #     cv2.putText(img, f"Fish-motion perception ON", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
                        #                 color=(255, 255, 255))

                        # if 100 < ts[time_i] < 160:
                        #     cv2.putText(img, f"Ground moving right", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                        #                 color=(255, 255, 255))
                        #
                        # if 180 < ts[time_i] < 240:
                        #     cv2.putText(img, f"Ground moving left", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                        #                 color=(255, 255, 255))

                        for j in range(x_particles.shape[1]):

                            dx = 20 * np.cos(np.radians(orientation_particles[time_i, j])) # 2 mm long fish
                            dy = 20 * np.sin(np.radians(orientation_particles[time_i, j]))

                            cv2.line(img, (int(x_particles[time_i, j] ), int(y_particles[time_i, j] )),
                                     (int(x_particles[time_i, j]  + dx ),
                                      int(y_particles[time_i, j]  + dy )),
                                     tuple(np.array(to_rgb(f"C{j % 10}")) * 255), 5, lineType=cv2.LINE_AA)

                            cv2.circle(img, (int(x_particles[time_i, j] ), int(y_particles[time_i, j] )), 8,
                                       tuple(np.array(to_rgb(f"C{j % 10}")) * 255), -1, lineType=cv2.LINE_AA)

                        writer.append_data(img)

                    writer.close()

                    # also make a plot of the trajectory
                    #fig = myfig.Figure(title=f"Example trajectory")
                    pl.figure(figsize=(10,10))
                    for fish_i in range(n):

                        pl.quiver(x_particles[:,fish_i][::20],
                                  y_particles[:,fish_i][::20],
                                  20 * np.cos(np.radians(orientation_particles[:, fish_i]))[::20],
                                  20 * np.sin(np.radians(orientation_particles[:, fish_i]))[::20],
                                  color = f'C{fish_i}',)
                    pl.xlim(0, 1024)
                    pl.ylim(0, 1024)
                    pl.savefig(output_path / f'collective_model_example_{genotype}.pdf')
                    pl.close()


                ang = np.deg2rad(np.arange(0, 360, 0.1))

                # Store this in a folder in the same style as Roy stores his behavior data.
                simple_data = dict({"body_angle": orientation_particles.T,
                                    "x_f": x_particles.T,
                                    "y_f": y_particles.T,
                                    "T": ts,
                                    "borders": np.array([np.cos(ang)*arena_size/2 + arena_size/2,
                                                np.sin(ang)*arena_size/2 + arena_size/2]).T})
                Header_new = dict({"ratio_pix_cm": [10]})

                output_path2 = output_path / genotype / f"5fish_{j}"
                output_path2.mkdir(parents=True, exist_ok=True)

                sio.savemat(output_path2 / "simple_data.mat", simple_data)
                sio.savemat(output_path2 / "Header_new.mat", Header_new)


