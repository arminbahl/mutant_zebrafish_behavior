import pandas as pd
from pathlib import Path
import pickle
import numpy as np

#root_path_setup0 = Path(r"Z:\arminbahl\Behavior Setup 0\free_swimming_4fish_data\dot_motion_coherence8_2")
#root_path_setup1 = Path(r"Z:\arminbahl\Behavior Setup 1\free_swimming_4fish_data\dot_motion_coherence8_2")

root_path_setup0 = Path("/n/home10/abahl/engert_lab_storage/arminbahl/Behavior Setup 0/free_swimming_4fish_data/dot_motion_coherence8_2")
root_path_setup1 = Path("/n/home10/abahl/engert_lab_storage/arminbahl/Behavior Setup 1/free_swimming_4fish_data/dot_motion_coherence8_2")

fishes_setup0 = [
    "2017_12_25_fish001",
    "2017_12_25_fish002",
    "2017_12_25_fish003",
    "2017_12_25_fish004",

    "2017_12_25_fish005",
    "2017_12_25_fish006",
    "2017_12_25_fish007",
    "2017_12_25_fish008",

    "2017_12_29_fish013",
    "2017_12_29_fish014",
    "2017_12_29_fish015",
    "2017_12_29_fish016",

    "2017_12_29_fish017",
    "2017_12_29_fish018",
    "2017_12_29_fish019",
    "2017_12_29_fish020",

    "2017_12_29_fish021",
    "2017_12_29_fish022",
    "2017_12_29_fish023",
    "2017_12_29_fish024",

    "2017_12_30_fish025",
    "2017_12_30_fish026",
    "2017_12_30_fish027",
    "2017_12_30_fish028",

    "2017_12_30_fish029",
    "2017_12_30_fish030",
    "2017_12_30_fish031",
    "2017_12_30_fish032",

]

fishes_setup1 = [
    "2017_12_25_fish001",
    "2017_12_25_fish002",
    "2017_12_25_fish003",
    "2017_12_25_fish004",

    "2017_12_25_fish005",
    "2017_12_25_fish006",
    "2017_12_25_fish007",
    "2017_12_25_fish008",

    "2017_12_29_fish009",
    "2017_12_29_fish010",
    "2017_12_29_fish011",
    "2017_12_29_fish012",

    "2017_12_29_fish013",
    "2017_12_29_fish014",
    "2017_12_29_fish015",
    "2017_12_29_fish016",

    "2017_12_29_fish017",
    "2017_12_29_fish018",
    "2017_12_29_fish019",
    "2017_12_29_fish020",

    "2017_12_30_fish021",
    "2017_12_30_fish022",
    "2017_12_30_fish023",
    "2017_12_30_fish024",

    "2017_12_30_fish025",
    "2017_12_30_fish026",
    "2017_12_30_fish027",
    "2017_12_30_fish028",

    "2017_12_30_fish029",
    "2017_12_30_fish030",
    "2017_12_30_fish031",
    "2017_12_30_fish032",
]

all_data = []
numtrials = 45

for setupID in [0, 1]:
    if setupID == 0:
        fishes = fishes_setup0
        root_path = root_path_setup0
    else:
        fishes = fishes_setup1
        root_path = root_path_setup1

    for i in range(len(fishes)):
        fish_ID = fishes[i]
        genotype = "wt"

        for trial in range(0, numtrials):
            print(fish_ID, genotype, trial)

            try:
                f = open(root_path / fish_ID / "raw_data" / f"trial{trial:03d}.dat", 'rb')
                data = pickle.load(f)
                f.close()
            except:
                break

            for stim in range(8):

                bout_times = data[f"bouts_start_stimulus_{stim:03d}"]["timestamp"]
                bout_xs = data[f"bouts_start_stimulus_{stim:03d}"]["fish_position_x"]
                bout_ys = data[f"bouts_start_stimulus_{stim:03d}"]["fish_position_y"]
                bout_start_fish_accumulated_orientation = data[f"bouts_start_stimulus_{stim:03d}"]["fish_accumulated_orientation"]
                bout_end_fish_accumulated_orientation = data[f"bouts_end_stimulus_{stim:03d}"]["fish_accumulated_orientation"]

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

df.to_hdf(root_path_setup0 / "all_data.h5", key="all_bouts", complevel=9)
