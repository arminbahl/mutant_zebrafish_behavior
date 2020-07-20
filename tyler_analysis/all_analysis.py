import numpy as np
import pandas as pd
import scipy.stats as stat
import json
from json import JSONEncoder

#############################################################################
################################### JSON ####################################
#############################################################################
def writeJSONAll():
    writeJSONOne()
    print("One finished.")
    writeJSONTwo()
    print("Two finished.")
    writeJSONThree()
    print("Three finished.")
    writeJSONFour()
    print("Four finished.")

def writeJSONOne():
    jsonData = {}
    jsonData['oneA'] = []
    jsonData['oneB'] = []
    jsonData['oneC'] = []

    for i in range(0, 10):
        # figOne
        jsonData['oneA'].append({'gen': 'wt', 'fileIndex': i, 'output': oneA("wt", i)})
        jsonData['oneA'].append({'gen': 'het', 'fileIndex': i, 'output': oneA("het", i)})
        jsonData['oneA'].append({'gen': 'hom', 'fileIndex': i, 'output': oneA("hom", i)})

        jsonData['oneB'].append({'gen': 'wt', 'fileIndex': i, 'output': oneB("wt", i)})
        jsonData['oneB'].append({'gen': 'het', 'fileIndex': i, 'output': oneB("het", i)})
        jsonData['oneB'].append({'gen': 'hom', 'fileIndex': i, 'output': oneB("hom", i)})

        jsonData['oneC'].append({'gen': 'wt', 'fileIndex': i, 'output': oneC("wt", i)})
        jsonData['oneC'].append({'gen': 'het', 'fileIndex': i, 'output': oneC("het", i)})
        jsonData['oneC'].append({'gen': 'hom', 'fileIndex': i, 'output': oneC("hom", i)})

    open('analysis_output/analysis_output1.txt', 'w').close()  # Erase contents of txt file
    with open('analysis_output/analysis_output1.txt', 'w') as outfile:
        json.dump(jsonData, outfile, cls=NumpyArrayEncoder)

def writeJSONTwo():
    jsonData = {}
    jsonData['twoAa'] = []
    jsonData['twoAb'] = []
    jsonData['twoB'] = []
    jsonData['twoC'] = []
    jsonData['twoD'] = []

    for i in range(0, 10):
        # figTwo
        jsonData['twoAa'].append({'gen': 'wt', 'fileIndex': i, 'output': twoAa("wt", i)})
        jsonData['twoAa'].append({'gen': 'het', 'fileIndex': i, 'output': twoAa("het", i)})
        jsonData['twoAa'].append({'gen': 'hom', 'fileIndex': i, 'output': twoAa("hom", i)})

        jsonData['twoAb'].append({'gen': 'wt', 'fileIndex': i, 'output': twoAb("wt", i)})
        jsonData['twoAb'].append({'gen': 'het', 'fileIndex': i, 'output': twoAb("het", i)})
        jsonData['twoAb'].append({'gen': 'hom', 'fileIndex': i, 'output': twoAb("hom", i)})

        jsonData['twoB'].append({'gen': 'wt', 'fileIndex': i, 'output': twoB("wt", i)})
        jsonData['twoB'].append({'gen': 'het', 'fileIndex': i, 'output': twoB("het", i)})
        jsonData['twoB'].append({'gen': 'hom', 'fileIndex': i, 'output': twoB("hom", i)})

        jsonData['twoC'].append({'gen': 'wt', 'fileIndex': i, 'output': twoC("wt", i)})
        jsonData['twoC'].append({'gen': 'het', 'fileIndex': i, 'output': twoC("het", i)})
        jsonData['twoC'].append({'gen': 'hom', 'fileIndex': i, 'output': twoC("hom", i)})

        jsonData['twoD'].append({'gen': 'wt', 'fileIndex': i, 'output': twoD("wt", i)})
        jsonData['twoD'].append({'gen': 'het', 'fileIndex': i, 'output': twoD("het", i)})
        jsonData['twoD'].append({'gen': 'hom', 'fileIndex': i, 'output': twoD("hom", i)})

    open('analysis_output/analysis_output2.txt', 'w').close()  # Erase contents of txt file
    with open('analysis_output/analysis_output2.txt', 'w') as outfile:
        json.dump(jsonData, outfile, cls=NumpyArrayEncoder)

def writeJSONThree():
    jsonData = {}
    jsonData['threeA'] = []
    jsonData['threeBC'] = []
    jsonData['threeD'] = []

    for i in range(0, 10):
        # figThree
        jsonData['threeA'].append({'gen': 'wt', 'fileIndex': i, 'output': threeA("wt", i)})
        jsonData['threeA'].append({'gen': 'het', 'fileIndex': i, 'output': threeA("het", i)})
        jsonData['threeA'].append({'gen': 'hom', 'fileIndex': i, 'output': threeA("hom", i)})

        jsonData['threeBC'].append({'gen': 'wt', 'fileIndex': i, 'output': threeBC("wt", i)})
        jsonData['threeBC'].append({'gen': 'het', 'fileIndex': i, 'output': threeBC("het", i)})
        jsonData['threeBC'].append({'gen': 'hom', 'fileIndex': i, 'output': threeBC("hom", i)})

        jsonData['threeD'].append({'gen': 'wt', 'fileIndex': i, 'output': threeD("wt", i)})
        jsonData['threeD'].append({'gen': 'het', 'fileIndex': i, 'output': threeD("het", i)})
        jsonData['threeD'].append({'gen': 'hom', 'fileIndex': i, 'output': threeD("hom", i)})

    open('analysis_output/analysis_output3.txt', 'w').close()  # Erase contents of txt file
    with open('analysis_output/analysis_output3.txt', 'w') as outfile:
        json.dump(jsonData, outfile, cls=NumpyArrayEncoder)

def writeJSONFour():
    jsonData = {}
    jsonData['fourABC'] = []
    jsonData['fourDE'] = []

    for i in range(0, 10):
        # figFour
        jsonData['fourABC'].append({'gen': 'wt', 'fileIndex': i, 'output': fourABC("wt", i)})
        jsonData['fourABC'].append({'gen': 'het', 'fileIndex': i, 'output': fourABC("het", i)})
        jsonData['fourABC'].append({'gen': 'hom', 'fileIndex': i, 'output': fourABC("hom", i)})

        jsonData['fourDE'].append({'gen': 'wt', 'fileIndex': i, 'output': fourDE("wt", i)})
        jsonData['fourDE'].append({'gen': 'het', 'fileIndex': i, 'output': fourDE("het", i)})
        jsonData['fourDE'].append({'gen': 'hom', 'fileIndex': i, 'output': fourDE("hom", i)})

    open('analysis_output/analysis_output4.txt', 'w').close()  # Erase contents of txt file
    with open('analysis_output/analysis_output4.txt', 'w') as outfile:
        json.dump(jsonData, outfile, cls=NumpyArrayEncoder)


#############################################################################
################################## PLOT 1 ###################################
#############################################################################

# 1A: Swims per second as a function of coherence (wt and gen)
def oneA(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    swims_per_second_list = np.zeros(4)
    swims_per_second_sem_list = np.zeros(4)
    for s in range(4):
        all_events = all_bouts.query("genotype=='" + gen + "' and 10<bout_time<20 and stim==" + str(s))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        fish_values = []
        for fish in fish_IDs:
            fish_events = all_events.query("fish_ID=='"+str(fish)+"'")
            trials = len(fish_events.index.get_level_values('trial').unique().values)*2
            fish_values.append(len(fish_events)/(10*trials))
        swims_per_second_list[s] = np.mean(fish_values)
        swims_per_second_sem_list[s] = stat.sem(fish_values, nan_policy = 'omit')
    return(gen,[swims_per_second_list,swims_per_second_sem_list])

# 1B: Fraction of correct bouts as a function of coherence (wt and gen)
def oneB(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    fraction_correct_list = np.zeros(4)
    fraction_correct_sem_list = np.zeros(4)
    for s in range(4):
        all_events = all_bouts.query("genotype=='"+gen+"' and 10<bout_time<20 and stim=="+str(s))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        fish_values = []
        for fish in fish_IDs:
            fish_events = all_events.query("fish_ID=='" + str(fish) + "'")
            fish_correct_events = fish_events.query("heading_angle_change>0")
            fish_values.append(100*len(fish_correct_events)/len(fish_events))
        fraction_correct_list[s] = np.mean(fish_values)
        fraction_correct_sem_list[s] = stat.sem(fish_values, nan_policy = 'omit')
    return(gen,[fraction_correct_list, fraction_correct_sem_list])

# 1C: Angle turned per correct bout as a function of coherence (wt and gen)
def oneC(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    average_correct_angle_list = np.zeros(4)
    average_correct_angle_sem_list = np.zeros(4)
    for s in range(4):
        all_events = all_bouts.query("genotype=='"+gen+"' and 10<bout_time<20 and stim=="+str(s))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        fish_values = []
        for fish in fish_IDs:
            fish_events = all_events.query("fish_ID=='" + str(fish) + "'")
            fish_correct_events = fish_events.query("heading_angle_change>0")
            fish_values.append(fish_correct_events["heading_angle_change"].mean())
        fish_values_cleaned = [x for x in fish_values if str(x) != 'nan']
        average_correct_angle_list[s] = np.mean(fish_values_cleaned)
        average_correct_angle_sem_list[s] = stat.sem(fish_values, nan_policy = 'omit')
    return(gen,[average_correct_angle_list, average_correct_angle_sem_list])


#############################################################################
################################## PLOT 2 ###################################
#############################################################################

# 2Aa: Probability correct as a function of coherence (wt and gen)
def twoAa(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    fraction_correct_list = np.zeros(4)
    fraction_correct_sem_list = np.zeros(4)
    for s in range(4):
        all_events = all_bouts.query("genotype=='" + gen + "' and 10<bout_time<20 and stim==" + str(s))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        fish_values = []
        for fish in fish_IDs:
            fish_events = all_events.query("fish_ID=='" + str(fish) + "'")
            fish_correct_events = fish_events.query("heading_angle_change>0")
            fish_values.append(100 * len(fish_correct_events) / len(fish_events))
        fraction_correct_list[s] = np.mean(fish_values)
        fraction_correct_sem_list[s] = stat.sem(fish_values, nan_policy = 'omit')
    return (gen, [fraction_correct_list, fraction_correct_sem_list])

# 2Ab: Interbout interval as a function of coherence (wt and gen)
def twoAb(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    interbout_interval_mean_list = np.zeros(4)
    interbout_interval_sem_list = np.zeros(4)
    for s in range(4):
        all_events = all_bouts.query("genotype=='" + gen + "' and 10<bout_time<20 and stim==" + str(s))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        fish_values = []
        for fish in fish_IDs:
            fish_events = all_events.query("fish_ID=='" + str(fish) + "'")
            fish_values.append(np.mean(fish_events["inter_bout_interval"]))
        interbout_interval_mean_list[s] = np.mean(fish_values)
        interbout_interval_sem_list[s] = stat.sem(fish_values, nan_policy = 'omit')
    return (gen, [interbout_interval_mean_list, interbout_interval_sem_list])

# 2B: Time-binned probability correct (wt or gen)
def twoB(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    step = 2
    start_time = 8
    end_time = 24

    time_binned_prob_correct_list = [[0] * int((end_time - start_time) / step)] * 4
    time_binned_prob_correct_sem_list = [[0] * int((end_time - start_time) / step)] * 4
    for s in range(4):
        time_bins = list(range(start_time,end_time,step))
        time_binned_mean = np.zeros(len(time_bins))
        time_binned_sem = np.zeros(len(time_bins))
        for i in range(0,len(time_bins)):
            all_events = all_bouts.query("genotype=='"+gen+"' and "+str(time_bins[i])+"<=bout_time<"+str(step+time_bins[i])+" and stim=="+str(s))
            fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
            fish_values = []
            for fish in fish_IDs:
                fish_events = all_events.query("fish_ID=='"+fish+"'")
                fish_correct_events = fish_events.query("heading_angle_change>0")
                fish_values.append(100 * len(fish_correct_events) / len(fish_events))
            time_binned_mean[i] = np.mean(fish_values)
            time_binned_sem[i] = stat.sem(fish_values, nan_policy = 'omit')
        time_binned_prob_correct_list[s] = time_binned_mean
        time_binned_prob_correct_sem_list[s] = time_binned_sem
    return(gen,[time_binned_prob_correct_list,time_binned_prob_correct_sem_list])

# 2C: Probability correct of consecutive bouts after stimulus start and end (wt or gen)
def twoC(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    beforeNum = 5
    afterNum = 3
    totalNum = beforeNum+afterNum

    consecutive_bouts_correct_prob = [[0]*totalNum]*4
    consecutive_bouts_correct_prob_sem = [[0]*totalNum]*4

    bad_fish = []
    for s in range(4):
        all_events = all_bouts.query("genotype=='" + gen + "' and 10<bout_time and stim==" + str(s))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        all_fish_values = []
        for fish in fish_IDs:
            if fish in bad_fish:
                continue
            fish_events = all_events.query("fish_ID=='" + fish + "'")
            fish_trials = len(fish_events.index.get_level_values("trial").unique().values)
            final_fish_trials = fish_trials
            individual_fish_values = np.zeros(totalNum)
            for trialNum in range(0, fish_trials):
                if fish_events.query("trial==" + str(trialNum)).empty:
                    final_fish_trials -= 1
                    continue
                fish_bout_times1 = fish_events.query("trial==" + str(trialNum))["bout_time"]
                fish_bout_times2 = fish_events.query("trial==" + str(trialNum) + "and bout_time>20")["bout_time"]
                if len(fish_bout_times1) < beforeNum:
                    final_fish_trials -= 1
                    continue
                if len(fish_bout_times2) < afterNum:
                    final_fish_trials -= 1
                    continue
                for i in range(0, beforeNum):
                    if fish_events.query("trial==" + str(trialNum) + "& bout_time==" + str(fish_bout_times1[i]))[
                        "heading_angle_change"][0] > 0:
                        individual_fish_values[i] += 1
                for i in range(0, afterNum):
                    if fish_events.query("trial==" + str(trialNum) + "& bout_time==" + str(fish_bout_times2[i]))[
                        "heading_angle_change"][0] > 0:
                        individual_fish_values[i + beforeNum] += 1
            if final_fish_trials < 5:
                print(fish)
                bad_fish.append(fish)
                continue
            if final_fish_trials > 5:
                all_fish_values.append([100 * x / final_fish_trials for x in individual_fish_values])
        consecutive_bouts_correct_prob[s] = np.mean(all_fish_values, axis=0)
        consecutive_bouts_correct_prob_sem[s] = stat.sem(all_fish_values, axis=0, nan_policy='omit')
    return (gen, [consecutive_bouts_correct_prob, consecutive_bouts_correct_prob_sem])

# 2D: Probability to swim in same direction as a function of interbout interval (0% coherence, wt and gen)
def twoD(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    step = 0.5
    max_time = 3
    time_bins = np.arange(0, max_time, step)

    same_direction_prob_list = np.zeros(len(time_bins))
    same_direction_prob_sem_list = np.zeros(len(time_bins))
    for i in range(0, len(time_bins)):
        all_events = all_bouts.query("genotype=='" + gen + "' and stim==0 and "+str(time_bins[i])+"<=inter_bout_interval<"+str(time_bins[i]+step))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        fish_values = []
        for fish in fish_IDs:
            fish_events = all_events.query("fish_ID=='" + str(fish) + "'")
            fish_values.append(100*sum(fish_events["same_as_previous"])/len(fish_events))
        same_direction_prob_list[i] = np.mean(fish_values)
        same_direction_prob_sem_list[i] = stats.sem(fish_values, nan_policy='omit')
    return (gen, [same_direction_prob_list, same_direction_prob_sem_list])

#############################################################################
################################## PLOT 3 ###################################
#############################################################################

# 3A Probability correct as a function of delay (interbout interval) for all stimulus levels (wt or gen)
def threeA(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    step = 0.5
    total_time = 2
    delay_binned_prob_correct_list = [[0] * int(total_time / step)] * 4
    delay_binned_prob_correct_sem_list = [[0] * int(total_time / step)] * 4
    for s in range(4):
        delay_bins = np.linspace(0, total_time, int(total_time / step), endpoint=False)
        delay_binned_mean = np.zeros(len(delay_bins))
        delay_binned_sem = np.zeros(len(delay_bins))
        for i in range(0,len(delay_bins)):
            all_events = all_bouts.query("genotype=='"+gen+"' and 10<bout_time<20 and "+str(delay_bins[i])+"<=inter_bout_interval<"+str(step+delay_bins[i])+" and stim=="+str(s))
            fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
            fish_values = []
            for fish in fish_IDs:
                fish_events = all_events.query("fish_ID=='"+fish+"'")
                fish_correct_events = fish_events.query("heading_angle_change>0")
                fish_values.append(100 * len(fish_correct_events) / len(fish_events))
            delay_binned_mean[i] = np.mean(fish_values)
            delay_binned_sem[i] = stat.sem(fish_values, nan_policy='omit')
        delay_binned_prob_correct_list[s] = delay_binned_mean
        delay_binned_prob_correct_sem_list[s] = delay_binned_sem
    return(gen,[delay_binned_prob_correct_list, delay_binned_prob_correct_sem_list])

# 3BC Angle turned per correct/incorrect bout as a function of delay for all stimulus levels (wt or gen)
def threeBC(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    step = 0.5
    total_time = 2
    delay_binned_correct_angle_list = [[0] * int(total_time / step)] * 4
    delay_binned_correct_angle_sem_list = [[0] * int(total_time / step)] * 4
    delay_binned_incorrect_angle_list = [[0] * int(total_time / step)] * 4
    delay_binned_incorrect_angle_sem_list = [[0] * int(total_time / step)] * 4
    for s in range(4):
        delay_bins = np.linspace(0, total_time, int(total_time / step), endpoint=False)
        correct_delay_binned_mean = np.zeros(len(delay_bins))
        correct_delay_binned_sem = np.zeros(len(delay_bins))
        incorrect_delay_binned_mean = np.zeros(len(delay_bins))
        incorrect_delay_binned_sem = np.zeros(len(delay_bins))
        for i in range(0, len(delay_bins)):
            all_events = all_bouts.query("genotype=='"+gen+"' and 10<bout_time<20 and "+str(delay_bins[i])+"<=inter_bout_interval<"+str(step+delay_bins[i])+" and stim=="+str(s))
            fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
            correct_fish_values = []
            incorrect_fish_values = []
            for fish in fish_IDs:
                fish_events = all_events.query("fish_ID=='" + fish + "'")
                if len(fish_events.query("heading_angle_change>0")["heading_angle_change"]) > 0:
                    correct_fish_values.append(np.mean(fish_events.query("heading_angle_change>0")["heading_angle_change"]))
                if len(fish_events.query("heading_angle_change<0")["heading_angle_change"]) > 0:
                    incorrect_fish_values.append(np.mean(fish_events.query("heading_angle_change<0")["heading_angle_change"]))
            correct_delay_binned_mean[i] = np.mean(correct_fish_values)
            correct_delay_binned_sem[i] = stat.sem(correct_fish_values, nan_policy='omit')
            incorrect_delay_binned_mean[i] = np.mean(incorrect_fish_values)
            incorrect_delay_binned_sem[i] = stat.sem(incorrect_fish_values, nan_policy='omit')
        delay_binned_correct_angle_list[s] = correct_delay_binned_mean
        delay_binned_correct_angle_sem_list[s] = correct_delay_binned_sem
        delay_binned_incorrect_angle_list[s] = [np.abs(x) for x in incorrect_delay_binned_mean]
        delay_binned_incorrect_angle_sem_list[s] = incorrect_delay_binned_sem
    return (gen, [delay_binned_correct_angle_list, delay_binned_correct_angle_sem_list], [delay_binned_incorrect_angle_list, delay_binned_incorrect_angle_sem_list])

# 3Dab Time for first bout and first correct bout after start of stimulus (wt and gen)
def threeD(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    first_bout_list = np.zeros(4)
    first_bout_sem_list = np.zeros(4)
    first_correct_bout_list = np.zeros(4)
    first_correct_bout_sem_list = np.zeros(4)
    for s in range(4):
        all_events = all_bouts.query("genotype=='" + gen + "' and 10<bout_time and stim==" + str(s))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        all_fish_first_bout_time = []
        all_fish_first_correct_bout_time = []
        for fish in fish_IDs:
            fish_events = all_events.query("fish_ID=='" + fish + "'")
            fish_trials = len(fish_events.index.get_level_values("trial").unique().values)
            individual_fish_first_vals = []
            individual_fish_first_correct_vals = []
            for trialNum in range(0, fish_trials):
                fish_trial_events = fish_events.query("trial==" + str(trialNum))
                if fish_trial_events.empty:
                    continue
                if fish_trial_events["bout_time"][0]>20:
                    continue
                if len(fish_trial_events)<1:
                    continue
                individual_fish_first_vals.append(fish_trial_events["bout_time"][0])
                for i in range(0, len(fish_trial_events)):
                    if fish_trial_events["heading_angle_change"][i] > 0:
                        if fish_trial_events["bout_time"][i] < fish_trial_events["bout_time"][0]:
                            continue
                        individual_fish_first_correct_vals.append(fish_trial_events["bout_time"][i])
                        break
            if len(individual_fish_first_vals) > 0:
                all_fish_first_bout_time.append(np.mean([n - 10 for n in individual_fish_first_vals]))
            if len(individual_fish_first_correct_vals) > 0:
                all_fish_first_correct_bout_time.append(np.mean([n - 10 for n in individual_fish_first_correct_vals]))
        first_bout_list[s] = np.mean(all_fish_first_bout_time)
        first_bout_sem_list[s] = stat.sem(all_fish_first_bout_time, nan_policy='omit')
        first_correct_bout_list[s] = np.mean(all_fish_first_correct_bout_time)
        first_correct_bout_sem_list[s] = stat.sem(all_fish_first_correct_bout_time, nan_policy='omit')

    return (gen, [first_bout_list, first_bout_sem_list], [first_correct_bout_list, first_correct_bout_sem_list])


#############################################################################
################################## PLOT 4 ###################################
#############################################################################

# 4ABC Average distance travelled per all (Aa)/correct (Ab)/incorrect (Ac) bout in any (A)/ x (B) / y (C) direction for all stimulus levels
def fourABC(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    average_distance_list = np.zeros(4)
    average_distance_sem_list = np.zeros(4)
    average_correct_distance_list = np.zeros(4)
    average_correct_distance_sem_list = np.zeros(4)
    average_incorrect_distance_list = np.zeros(4)
    average_incorrect_distance_sem_list = np.zeros(4)
    average_correct_x_distance_list = np.zeros(4)
    average_correct_x_distance_sem_list = np.zeros(4)
    average_incorrect_x_distance_list = np.zeros(4)
    average_incorrect_x_distance_sem_list = np.zeros(4)
    average_correct_y_distance_list = np.zeros(4)
    average_correct_y_distance_sem_list = np.zeros(4)
    average_incorrect_y_distance_list = np.zeros(4)
    average_incorrect_y_distance_sem_list = np.zeros(4)
    for s in range(4):
        all_events = all_bouts.query("genotype=='" + gen + "' and 10<bout_time<20 and stim==" + str(s))
        fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
        all_fish_values = []
        all_fish_correct_values = []
        all_fish_incorrect_values = []
        all_fish_correct_x_values = []
        all_fish_incorrect_x_values = []
        all_fish_correct_y_values = []
        all_fish_incorrect_y_values = []
        for fish in fish_IDs:
            fish_events = all_events.query("fish_ID=='" + str(fish) + "'")
            individual_fish_values = []
            individual_fish_correct_values = []
            individual_fish_incorrect_values = []
            individual_fish_correct_x_values = []
            individual_fish_incorrect_x_values = []
            individual_fish_correct_y_values = []
            individual_fish_incorrect_y_values = []
            for i in range(0, len(fish_events)-1):
                if fish_events.index.get_level_values("trial")[i] == fish_events.index.get_level_values("trial")[i+1] and fish_events["bout_time"][i] < fish_events["bout_time"][i+1]:
                    dist = np.sqrt((fish_events["bout_x"][i + 1] - fish_events["bout_x"][i]) ** 2 + (fish_events["bout_y"][i + 1] - fish_events["bout_y"][i]) ** 2)
                    individual_fish_values.append(dist)
                    if fish_events["heading_angle_change"][i] > 0:
                        individual_fish_correct_values.append(dist)
                        individual_fish_correct_x_values.append(np.sin(np.deg2rad(fish_events["heading_angle_change"][i])) * dist)
                        individual_fish_correct_y_values.append(np.cos(np.deg2rad(fish_events["heading_angle_change"][i])) * dist)
                    else:
                        individual_fish_incorrect_values.append(dist)
                        individual_fish_incorrect_x_values.append(np.sin(np.deg2rad(fish_events["heading_angle_change"][i])) * dist)
                        individual_fish_incorrect_y_values.append(np.cos(np.deg2rad(fish_events["heading_angle_change"][i])) * dist)
            if individual_fish_values != []:
                all_fish_values.append(np.mean(individual_fish_values)*6)
            if individual_fish_correct_values != []:
                all_fish_correct_values.append(np.mean(individual_fish_correct_values) * 6)
            if individual_fish_incorrect_values != []:
                all_fish_incorrect_values.append(np.mean(individual_fish_incorrect_values) * 6)
            if individual_fish_correct_x_values != []:
                all_fish_correct_x_values.append(np.mean(individual_fish_correct_x_values) * 6)
            if individual_fish_incorrect_x_values != []:
                all_fish_incorrect_x_values.append(np.abs(np.mean(individual_fish_incorrect_x_values)) * 6)
            if individual_fish_correct_y_values != []:
                all_fish_correct_y_values.append(np.mean(individual_fish_correct_y_values) * 6)
            if individual_fish_incorrect_y_values != []:
                all_fish_incorrect_y_values.append(np.abs(np.mean(individual_fish_incorrect_y_values)) * 6)
        average_distance_list[s] = np.mean(all_fish_values)
        average_distance_sem_list[s] = stats.sem(all_fish_values, nan_policy='omit')
        average_correct_distance_list[s] = np.mean(all_fish_correct_values)
        average_correct_distance_sem_list[s] = stats.sem(all_fish_correct_values, nan_policy='omit')
        average_incorrect_distance_list[s] = np.mean(all_fish_incorrect_values)
        average_incorrect_distance_sem_list[s] = stats.sem(all_fish_incorrect_values, nan_policy='omit')
        average_correct_x_distance_list[s] = np.mean(all_fish_correct_x_values)
        average_correct_x_distance_sem_list[s] = stats.sem(all_fish_correct_x_values, nan_policy='omit')
        average_incorrect_x_distance_list[s] = np.mean(all_fish_incorrect_x_values)
        average_incorrect_x_distance_sem_list[s] = stats.sem(all_fish_incorrect_x_values, nan_policy='omit')
        average_correct_y_distance_list[s] = np.mean(all_fish_correct_y_values)
        average_correct_y_distance_sem_list[s] = stats.sem(all_fish_correct_y_values, nan_policy='omit')
        average_incorrect_y_distance_list[s] = np.mean(all_fish_incorrect_y_values)
        average_incorrect_y_distance_sem_list[s] = stats.sem(all_fish_incorrect_y_values, nan_policy='omit')
    return (gen, [average_distance_list, average_distance_sem_list], [average_correct_distance_list, average_correct_distance_sem_list], [average_incorrect_distance_list, average_incorrect_distance_sem_list], [average_correct_x_distance_list, average_correct_x_distance_sem_list], [average_incorrect_x_distance_list, average_incorrect_x_distance_sem_list], [average_correct_y_distance_list, average_correct_y_distance_sem_list], [average_incorrect_y_distance_list, average_incorrect_y_distance_sem_list])

# 4DE Average distance travelled per correct/incorrect bout for all stimulus levels but time binned
def fourDE(gen, fileIndex):
    all_bouts = readHDF(fileIndex)

    step = 5
    start_time = 5
    end_time = 25

    time_binned_correct_bout_dist_list = [[0] * int((end_time - start_time) / step)] * 4
    time_binned_correct_bout_dist_sem_list = [[0] * int((end_time - start_time) / step)] * 4
    time_binned_incorrect_bout_dist_list = [[0] * int((end_time - start_time) / step)] * 4
    time_binned_incorrect_bout_dist_sem_list = [[0] * int((end_time - start_time) / step)] * 4
    for s in range(4):
        time_bins = list(range(start_time, end_time, step))
        time_binned_correct_mean = np.zeros(len(time_bins))
        time_binned_correct_sem = np.zeros(len(time_bins))
        time_binned_incorrect_mean = np.zeros(len(time_bins))
        time_binned_incorrect_sem = np.zeros(len(time_bins))
        for i in range(0,len(time_bins)):
            all_events = all_bouts.query("genotype=='"+gen+"' and "+str(time_bins[i])+"<=bout_time<"+str(step+time_bins[i])+" and stim=="+str(s))
            fish_IDs = all_events.index.get_level_values('fish_ID').unique().values
            all_fish_correct_distance = []
            all_fish_incorrect_distance = []
            for fish in fish_IDs:
                fish_events = all_events.query("fish_ID=='"+fish+"'")
                individual_fish_correct_distance = []
                individual_fish_incorrect_distance = []
                for j in range(0, len(fish_events)-1):
                    if fish_events.index.get_level_values("trial")[j] == fish_events.index.get_level_values("trial")[j + 1] and fish_events["bout_time"][j] < fish_events["bout_time"][j + 1]:
                        dist = np.sqrt((fish_events["bout_x"][j + 1] - fish_events["bout_x"][j]) ** 2 + (fish_events["bout_y"][j + 1] - fish_events["bout_y"][j]) ** 2)
                        if fish_events["heading_angle_change"][j] > 0:
                            individual_fish_correct_distance.append(dist)
                        elif fish_events["heading_angle_change"][j] < 0:
                            individual_fish_incorrect_distance.append(dist)
                if individual_fish_correct_distance != []:
                    all_fish_correct_distance.append(np.mean(individual_fish_correct_distance) * 6)
                if individual_fish_incorrect_distance != []:
                    all_fish_incorrect_distance.append(np.mean(individual_fish_incorrect_distance) * 6)
            time_binned_correct_mean[i] = np.mean(all_fish_correct_distance)
            time_binned_correct_sem[i] = stats.sem(all_fish_correct_distance, nan_policy='omit')
            time_binned_incorrect_mean[i] = np.mean(all_fish_incorrect_distance)
            time_binned_incorrect_sem[i] = stats.sem(all_fish_incorrect_distance, nan_policy='omit')
        time_binned_correct_bout_dist_list[s] = time_binned_correct_mean
        time_binned_correct_bout_dist_sem_list[s] = time_binned_correct_sem
        time_binned_incorrect_bout_dist_list[s] = time_binned_incorrect_mean
        time_binned_incorrect_bout_dist_sem_list[s] = time_binned_incorrect_sem
    return(gen, [time_binned_correct_bout_dist_list, time_binned_correct_bout_dist_sem_list], [time_binned_incorrect_bout_dist_list, time_binned_incorrect_bout_dist_sem_list])

#############################################################################
################################### MISC ####################################
#############################################################################
def readHDF(fileIndex):
    if fileIndex == 0:  # Access disc1_hetinx data
        all_bouts = pd.read_hdf("data/disc1_hetinx/all_data.h5", key="all_bouts")
    elif fileIndex == 1:  # Access immp2l_NIBR data
        all_bouts = pd.read_hdf("data/immp2l_NIBR/all_data.h5", key="all_bouts")
    elif fileIndex == 2:  # Access immp2l_summer data
        all_bouts = pd.read_hdf("data/immp2l_summer/all_data.h5", key="all_bouts")
    elif fileIndex == 3:  # Access scn1lab_NIBR data
        all_bouts = pd.read_hdf("data/scn1lab_NIBR/all_data.h5", key="all_bouts")
    elif fileIndex == 4:  # Access scn1lab_sa16474 data
        all_bouts = pd.read_hdf("data/scn1lab_sa16474/all_data.h5", key="all_bouts")
    elif fileIndex == 5:  # Access surrogate_fish1 data
        all_bouts = pd.read_hdf("data/surrogate_fish1/all_data.h5", key="all_bouts")
    elif fileIndex == 6:  # Access surrogate_fish2 data
        all_bouts = pd.read_hdf("data/surrogate_fish2/all_data.h5", key="all_bouts")
    elif fileIndex == 6.5:  # Access surrogate_fish2 data
        all_bouts = pd.read_hdf("data/surrogate_fish2/all_data_best_model_repeat4.h5", key="all_bouts")
    elif fileIndex == 7:  # Access surrogate_fish3 data
        all_bouts = pd.read_hdf("data/surrogate_fish3/all_data.h5", key="all_bouts")
    elif fileIndex == 8: # Access scn1lab_NIBR_20200708 data
        all_bouts = pd.read_hdf("data/scn1lab_NIBR_20200708/all_data.h5", key="all_bouts")
    elif fileIndex == 9: # Access scn1lab_zirc_20200710 data
        all_bouts = pd.read_hdf("data/scn1lab_zirc_20200710/all_data.h5", key="all_bouts")
    return(all_bouts)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
