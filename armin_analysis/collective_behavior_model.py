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

@jit(nopython=True)
def determine_angle_between_fish(ang1, x1, y1, x2, y2):
    ang2 = np.degrees(np.arctan2(y2-y1, x2-x1))
    if ang2 < 0:
        ang2 = 360 + ang2

    if ang2 > ang1:
        ang2 -= 360

    angle_between_fish = (ang1 - ang2)

    return angle_between_fish
    #if angle_between_fish <= 180:
    #    return
    #print(angle_between_fish)


#determine_angle_between_fish(90, 0, 0, -0.1, -10)
#sdfsf
@jit(nopython=True)
def simulate_particles(genotype,
                       dt,
                       ts,
                       integrated_motion_perception_particles,
                       current_clutter_perception_particles,
                       x_particles,
                       y_particles,
                       orientation_particles,
                       turning_speed_particles,
                       forward_speed_particles):

    tau_forward_speed = 0.1  # s
    tau_turning_speed = 0.1  # s
    effect_strength_motion = 2
    effect_strength_clutter = 10 # negative is repulsive
    fish_size = 10  # 10 pixel (one body length); arena is 512 pixel
    fish_forward_speed = 10

    n_particles = integrated_motion_perception_particles.shape[1]

    # wt scn1lab_sa16474
    if genotype == 0:
        #tau_perception = 0.4  # s
        #perception_noise_sigma = 5.9
        #T = 0.4
        #p_below = 0.009098
        #p_above = 0.027

        tau = 0.529309
        sigma = 8.738211
        T = 0.778464
        p_below = 0.008042
        p_above = 0.03623

    # het scn1lab_sa16474
    if genotype == 1:
        #tau_perception = 1.4

        #perception_noise_sigma = 4
        #T = 0.203593
        #p_below = 0.001663
        #p_above = 0.019993

        tau=0.880916
        sigma=5.757828
        T=0.370462
        p_below=0.006409
        p_above=0.016526

    # wt scn1lab_NIBR
    if genotype == 2:
        tau=1.364913
        sigma=12.160690
        T=0.507576
        p_below=0.006358
        p_above=0.014497

    if genotype == 3:
        tau=1.634761
        sigma=6.743575
        T=0.236815
        p_below=0.004419
        p_above=0.008745

    # Disc WT
    if genotype == 4:
        tau=0.529309
        sigma=8.738211
        T=0.778464
        p_below=0.008042
        p_above=0.036234

    if genotype == 5:
        tau=0.529309
        sigma=8.738211
        T=0.778464
        p_below=0.008042
        p_above=0.036234

    if genotype == 6:
        tau=0.880916
        sigma=5.757828
        T=0.370462
        p_below=0.006409
        p_above=0.016526

    for time_i in range(1, ts.shape[0]):
        ts[time_i] = ts[time_i - 1] + dt

        for particle_i in range(n_particles):

            current_motion_perception_particle_i = 0
            current_clutter_perception_particle_i = 0

            # turn on percepttion between 20 and 60 s
            if 20 <= ts[time_i] < 80:

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

                    #print(angle_between_orientation_vectors)
                    #return
                    # is it on the right or the left side?
                    angle_between_fish = determine_angle_between_fish(ang1,
                                                    x_particles[time_i - 1, particle_i],
                                                    y_particles[time_i - 1, particle_i],
                                                    x_particles[time_i - 1, particle_j],
                                                    y_particles[time_i - 1, particle_j])


                    perpendicular_motion_drive = effect_strength_motion * \
                                                 forward_speed_particles[time_i - 1, particle_j] * \
                                                 np.sin(np.radians(angle_between_orientation_vectors)) / distance

                    # generally, fish are repulsive based on angle the other fish casts on the retina!

                    size_on_retina = 2*np.arctan(fish_size/(2*distance))

                    if 0 <= angle_between_fish < 180:
                        current_clutter_perception_particle_i += effect_strength_clutter*size_on_retina
                    if 180 <= angle_between_fish < 360:
                        current_clutter_perception_particle_i -= effect_strength_clutter*size_on_retina

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
            x_particles[time_i, particle_i] = x_particles[time_i, particle_i] % 512
            y_particles[time_i, particle_i] = y_particles[time_i, particle_i] % 512

            #
            # if x_particles[time_i, particle_i] >= 512:
            #     x_particles[time_i, particle_i] = 511 - (x_particles[time_i, particle_i] - 512)
            #     recompute_orientation = True
            #
            # if y_particles[time_i, particle_i] >= 512:
            #     y_particles[time_i, particle_i] = 511 - (y_particles[time_i, particle_i] - 512)
            #     recompute_orientation = True
            #
            # if x_particles[time_i, particle_i] < 0:
            #     x_particles[time_i, particle_i] = 0 - (x_particles[time_i, particle_i] - 0)
            #     recompute_orientation = True
            #
            # if y_particles[time_i, particle_i] < 0:
            #     y_particles[time_i, particle_i] = 0 - (y_particles[time_i, particle_i] - 0)
            #     recompute_orientation = True

            if recompute_orientation == True:
                orientation_particles[time_i, particle_i] = np.degrees(np.arctan2(y_particles[time_i, particle_i] - y_particles[time_i - 1, particle_i],
                                                                                  x_particles[time_i, particle_i] - x_particles[time_i - 1, particle_i]))

            orientation_particles[time_i, particle_i] = orientation_particles[time_i, particle_i] % 360

dt = 0.01
n = 10

# Compute the chance levels
x_particles = np.random.random(size=(int(100/dt), n))*512
y_particles = np.random.random(size=(int(100/dt), n))*512
orientation_particles = np.random.random(size=(int(100/dt), n))*360

# calculate chance distance
ds = []
for k in range(10):
    for l in range(10):
        if k == l:
            continue

        ds.append(np.sqrt((x_particles[:, k] - x_particles[:, l])**2 +
                          (y_particles[:, k] - y_particles[:, l])**2).mean())

chance_distance = np.mean(ds)

# Calculate polarizion chance
x = np.cos(np.radians(orientation_particles)).mean(axis=1)
y = np.sin(np.radians(orientation_particles)).mean(axis=1)

chance_polarization = np.sqrt(x**2 + y**2).mean()

for genotype in [0]:
    polarizations = []
    ds_over_time = []
    for j in range(200):
        print(j)

        ts = np.zeros(int(100/dt))
        integrated_motion_perception_particles = np.zeros((int(100/dt), n))
        current_clutter_perception_particles = np.zeros((int(100/dt), n))

        x_particles = np.random.random(size=(int(100/dt), n)) * 512
        y_particles = np.random.random(size=(int(100/dt), n)) * 512
        orientation_particles = np.random.random(size=(int(100/dt), n)) * 360
        forward_speed_particles = np.zeros((int(100/dt), n))
        turning_speed_particles = np.zeros((int(100/dt), n))

        simulate_particles(genotype,
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
            root_path = Path("/Users/arminbahl/Desktop")
            writer = imageio.get_writer(root_path / f'collective_model_genotype{genotype}.mp4', codec='libx264', fps=30,
                                        ffmpeg_params=["-b:v", "8M"])

            node_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

            for time_i in tqdm(np.arange(0, ts.shape[0] - 1, 20)):

                img = np.zeros((512, 512), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                for j in range(x_particles.shape[1]):

                    cv2.putText(img, f"{ts[time_i]:.1f} s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                color=(255, 255, 255))
                    if 20 < ts[time_i] < 80:
                        cv2.putText(img, f"Fish-motion perception ON", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                    color=(255, 255, 255))

                    if 100 < ts[time_i] < 160:
                        cv2.putText(img, f"Ground moving right", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                    color=(255, 255, 255))

                    if 180 < ts[time_i] < 240:
                        cv2.putText(img, f"Ground moving left", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                    color=(255, 255, 255))

                    dx = 10 * np.cos(np.radians(orientation_particles[time_i, j]))
                    dy = 10 * np.sin(np.radians(orientation_particles[time_i, j]))
                    scale = 1
                    cv2.line(img, (int(x_particles[time_i, j] * scale), int(y_particles[time_i, j] * scale)),
                             (int(x_particles[time_i, j] * scale + dx * scale),
                              int(y_particles[time_i, j] * scale + dy * scale)),
                             tuple(np.array(to_rgb(f"C{j % 10}")) * 255), 2, lineType=cv2.LINE_AA)

                    cv2.circle(img, (int(x_particles[time_i, j] * scale), int(y_particles[time_i, j] * scale)), 4,
                               tuple(np.array(to_rgb(f"C{j % 10}")) * 255), -1, lineType=cv2.LINE_AA)

                writer.append_data(img)

            writer.close()

        # Calculate polarizion
        x = np.cos(np.radians(orientation_particles)).mean(axis=1)
        y = np.sin(np.radians(orientation_particles)).mean(axis=1)

        polarizations.append(np.sqrt(x**2 + y**2))

        ds = []
        for k in range(10):
            for l in range(10):
                if k==l:
                    continue

                ds.append(np.sqrt((x_particles[:, k] - x_particles[:, l])**2 +
                                  (y_particles[:, k] - y_particles[:, l])**2))

        ds_over_time.append(np.mean(ds, axis=0))

    pl.figure(0)
    if genotype == 0:
        pl.plot(ts, np.mean(polarizations, axis=0), color=f"C0", label="SCN_SA16474: WT")
    if genotype == 1:
        pl.plot(ts, np.mean(polarizations, axis=0), color=f"C1", label="SCN_SA16474: het")
    if genotype == 2:
        pl.plot(ts, np.mean(polarizations, axis=0), color=f"C2", label="SCN_NIBR: WT")
    if genotype == 3:
        pl.plot(ts, np.mean(polarizations, axis=0), color=f"C3", label="SCN_NIBR: Het")
    if genotype == 4:
        pl.plot(ts, np.mean(polarizations, axis=0), color=f"C4", label="DISC: WT")
    if genotype == 5:
        pl.plot(ts, np.mean(polarizations, axis=0), color=f"C5", label="DISC: Het")

    pl.figure(1)
    if genotype == 0:
        pl.plot(ts, np.mean(ds_over_time, axis=0), color=f"C0", label=["SCN_SA16474: WT"])
    if genotype == 1:
        pl.plot(ts, np.mean(ds_over_time, axis=0), color=f"C1", label=["SCN_SA16474: het"])
    if genotype == 2:
        pl.plot(ts, np.mean(ds_over_time, axis=0), color=f"C2", label=["SCN_NIBR: WT"])
    if genotype == 3:
        pl.plot(ts, np.mean(ds_over_time, axis=0), color=f"C3", label=["SCN_NIBR: Het"])
    if genotype == 4:
        pl.plot(ts, np.mean(ds_over_time, axis=0), color=f"C4", label="DISC: WT")
    if genotype == 5:
        pl.plot(ts, np.mean(ds_over_time, axis=0), color=f"C5", label="DISC: Het")

pl.figure(0)
pl.axvspan(20, 80, lw=0, edgecolor='none', facecolor="gray", zorder=0, alpha=0.5)
#pl.axvspan(100, 160, lw=0, edgecolor='none', facecolor="gray", zorder=0, alpha=0.5)
#pl.axvspan(180, 240, lw=0, edgecolor='none', facecolor="gray", zorder=0, alpha=0.5)
pl.axhline(chance_polarization, color='black', label='Chance level')
pl.xlabel("Time (s)")
pl.ylabel("Polarization")
pl.ylim(0, 1)
pl.legend()

pl.figure(1)
pl.axvspan(20, 80, lw=0, edgecolor='none', facecolor="gray", zorder=0, alpha=0.5)
#pl.axvspan(100, 160, lw=0, edgecolor='none', facecolor="gray", zorder=0, alpha=0.5)
#pl.axvspan(180, 240, lw=0, edgecolor='none', facecolor="gray", zorder=0, alpha=0.5)
pl.axhline(chance_distance, color='black', label='Chance level')
pl.xlabel("Time (s)")
pl.ylim(170, 300)
pl.ylabel("Average neighbor distance (mm)")
pl.legend()

pl.show()

# pl.plot(x_particles[:,0], y_particles[:,0], '-')
# pl.plot(x_particles[:,1], y_particles[:,1], '-')
# pl.figure()
#
# pl.plot(orientation_particles[:, 0])
# pl.figure()
# pl.plot(particles_perception[:, 0])
# pl.plot(particles_perception[:, 1])
# pl.show()
# h
#
# for k in range(10):
#     x = np.cos(np.radians(orientation_particles[:, k]))*k
#     y = np.sin(np.radians(orientation_particles[:, k]))*k
#
#     pl.plot(x, y)

#pl.show()

#pl.plot(ts, forward_speed_particles[:, 0])
# #pl.show()
#
# pl.plot(ts[:1000], particles_perception[:1000,0])
# pl.plot(ts[1000:2600], particles_perception[1000:2600,0])
# pl.plot(ts[2600:], particles_perception[2600:,0])
#
# pl.figure()
# pl.plot(x_particles[:1000,0], y_particles[:1000,0], '-')
# pl.plot(x_particles[1000:2600,0], y_particles[1000:2600,0], '-')
# pl.plot(x_particles[2600:,0], y_particles[2600:,0], '-')
# pl.xlim(0, 512)
# pl.ylim(0, 512)
#
# pl.figure()
# pl.plot(x_particles[:,1], y_particles[:,1])
# pl.plot(x_particles[:,2], y_particles[:,2])
# pl.plot(x_particles[:,3], y_particles[:,3])
# pl.plot(x_particles[:,4], y_particles[:,4])
#
# pl.show()
# dsdfgh
# pl.figure()
# pl.plot(ts, forward_speed_particles[:,0])
# pl.figure()
# pl.plot(ts, turning_speed_particles[:,0])
# pl.show()
#
# pl.figure()
# pl.plot(ts, orientation_particles[:,0])
#
# pl.show()
# sdf
#
# pl.plot(x_particles[:, 0], y_particles[:, 0])
# pl.plot(x_particles[:, 1], y_particles[:, 1])
# #pl.show()
# #hjjjg
# print(x_particles[:, 0])
# print(y_particles[:, 0])

