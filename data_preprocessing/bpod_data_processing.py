import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import os
import utils.load_nested_structs as load_ns
import numpy as np
import pandas as pd
import math

def find_bpod_file(mouse, date, protocol_type):
    Bpod_data_path = "W:\\users\\Yvonne\\photometry_2AC\\bpod_data\\" + mouse + '\\' + protocol_type + '\\Session Data\\'
    #Bpod_data_path = 'C:\\Users\\francescag\\Documents\\PhD_Project\\SNL_photo_photometry\\bpod_data\\'+ mouse + '\\' + protocol_type + '\\Session Data\\'
    Bpod_file_search_tool = mouse + '_' + protocol_type + '_' + date
    files_in_bpod_path = os.listdir(Bpod_data_path)
    files_on_that_day = [s for s in files_in_bpod_path if Bpod_file_search_tool in s]
    mat_files_on_that_day = [s for s in files_on_that_day if '.mat' in s]

    if len(mat_files_on_that_day) == 2:
        no_extension_files = [os.path.splitext(filename)[0] for filename in mat_files_on_that_day]
        file_times = [filename.split('_')[-1] for filename in no_extension_files]
        main_session_file = Bpod_data_path + mat_files_on_that_day[file_times.index(max(file_times))]
        return main_session_file
    elif len(mat_files_on_that_day) == 1:
        no_extension_files = [os.path.splitext(filename)[0] for filename in mat_files_on_that_day]
        file_times = [filename.split('_')[-1] for filename in no_extension_files]
        main_session_file = Bpod_data_path + mat_files_on_that_day[file_times.index(max(file_times))]
        return main_session_file
    else:
        print('0 or more than 2 sessions that day!')

def find_daq_file(mouse, date):
    daq_data_path = "W:\\users\\Yvonne\\photometry_2AC\\freely_moving_photometry_data\\" + mouse + '\\'
    #daq_data_path ='C:\\Users\\francescag\\Documents\\PhD_Project\\SNL_photo_photometry\\freely_moving_photometry_data\\' + mouse + '\\'
    folders_in_photo_path = os.listdir(daq_data_path)
    folders_on_that_day = [s for s in folders_in_photo_path if date in s]

    if len(folders_on_that_day) == 2:
        main_session_file = daq_data_path + '/' + folders_on_that_day[-1] + '/' + 'AI.tdms'
        return main_session_file
        print('2 sessions that day')
    elif len(folders_on_that_day) == 1:
        main_session_file = daq_data_path + '/' + folders_on_that_day[-1] + '/' + 'AI.tdms'
        return main_session_file
    else:
        print('0 or more than 2 sessions that day!')



def load_bpod_file(main_session_file):
    # gets the Bpod data out of MATLAB struct and into python-friendly format
    loaded_bpod_file = load_ns.loadmat(main_session_file)

    # as RawEvents.Trial is a cell array of structs in MATLAB, we have to loop through the array and convert the structs to dicts
    trial_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

    for trial_num, trial in enumerate(trial_raw_events):
        trial_raw_events[trial_num] = load_ns._todict(trial)

    loaded_bpod_file['SessionData']['RawEvents']['Trial'] = trial_raw_events
    trial_settings = loaded_bpod_file['SessionData']['TrialSettings']
    first_trial = trial_raw_events[0]
    first_trial_TTL = first_trial['States']['TrialStart']
    return loaded_bpod_file, trial_raw_events


def find_num_times_in_state(trial_states):
    unique_states = np.unique(trial_states)
    state_occurences = np.zeros(trial_states.shape)
    max_occurences = np.zeros(trial_states.shape)
    for state in unique_states:
        total_occurences = np.where(trial_states==state)[0].shape[0]
        num_occurences = 0
        for idx, val in enumerate(trial_states):
            if val==state:
                num_occurences+=1
                state_occurences[idx] = num_occurences
                max_occurences[idx] = total_occurences
    return state_occurences, max_occurences


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq, clock_pulses):
    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
    original_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

    daq_trials_start_ttls = trial_start_ttls_daq
    # loops through all the trials and pulls out all the states
    for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):
        state_info = {}
        event_info = {}
        trial_states = original_state_data_all_trials[trial]
        num_states = (len(trial_states))
        state_info['Trial num'] = np.ones((num_states)) * trial
        state_info['Trial type'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['TrialSequence'][trial]
        state_info['State type'] = trial_states
        num_times_in_state = find_num_times_in_state(trial_states)
        state_info['Instance in state'] = num_times_in_state[0]
        state_info['Max times in state'] = num_times_in_state[1]
        state_info['State name'] = loaded_bpod_file['SessionData']['RawData']['OriginalStateNamesByNumber'][0][
            trial_states - 1]
        state_info['Time start'] = state_timestamps[0:-1] + daq_trials_start_ttls[trial]
        state_info['Time end'] = state_timestamps[1:] + daq_trials_start_ttls[trial]

        #added by Matt
        #state_info['Reward amount'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['SettingsFile']['GUI']['RewardAmount']
        #state_info['Punish'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['SettingsFile']['GUI']['Punish']
        #state_info['VariableCueDelay'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['SettingsFile']['GUI']['VariableCueDelay']


        loc_lastCueDelay = [i for i, state in enumerate(trial_states) if state == 3][-1]
        loc_WaitForPortOut = [i for i, state in enumerate(trial_states) if state == 4][0]
        TrialCueDelay = state_timestamps[loc_WaitForPortOut] - state_timestamps[loc_lastCueDelay]
        state_info['TrialCueDelay'] = np.ones((num_states)) * TrialCueDelay


        #state_info['Start camera frame'] = find_nearest(clock_pulses, state_info['Time start'])
        #state_info['End camera frame'] = find_nearest(clock_pulses, state_info['Time end'])
        state_info['Response'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['ChosenSide'][trial]
        if trial > 0:
            state_info['Last response'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['ChosenSide'][
                trial - 1]
            state_info['Last outcome'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['Outcomes'][trial - 1]
        else:
            state_info['Last response'] = np.ones((num_states)) * -1
            state_info['Last outcome'] = np.ones((num_states)) * -1
        state_info['Trial start'] = np.ones((num_states)) * daq_trials_start_ttls[trial]
        state_info['Trial end'] = np.ones((num_states)) * (state_timestamps[-1] + daq_trials_start_ttls[trial])
        state_info['Trial outcome'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['Outcomes'][trial]
        state_info['First response'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['FirstPoke'][trial]
        if hasattr(loaded_bpod_file['SessionData']['TrialSettings'][trial], 'RewardChangeBlock'):
            state_info['Reward block'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['TrialSettings'][trial].RewardChangeBlock

        if loaded_bpod_file['SessionData']['FirstPoke'][trial] == loaded_bpod_file['SessionData']['TrialSide'][trial]:
            state_info['First choice correct'] = np.ones(num_states)
            event_info['First choice correct'] = [1]
            event_info['Trial num'] = [trial]
            event_info['Trial type'] = [loaded_bpod_file['SessionData']['TrialSequence'][trial]]
            event_info['State type'] = [8.5]
            event_info['Instance in state'] = [1]
            event_info['Max times in state'] = [1]
            event_info['State name'] = ['Leaving reward port']
            if hasattr(loaded_bpod_file['SessionData']['TrialSettings'][trial], 'RewardChangeBlock'):
                event_info['Reward block'] = loaded_bpod_file['SessionData']['TrialSettings'][trial].RewardChangeBlock
            event_info['Response'] = [loaded_bpod_file['SessionData']['ChosenSide'][trial]]
            if trial > 0:
                event_info['Last response'] = [loaded_bpod_file['SessionData']['ChosenSide'][
                                                   trial - 1]]
                event_info['Last outcome'] = [loaded_bpod_file['SessionData']['Outcomes'][
                                                  trial - 1]]
            else:
                event_info['Last response'] = [-1]
                event_info['Last outcome'] = [-1]
            event_info['Trial outcome'] = [loaded_bpod_file['SessionData']['Outcomes'][trial]]
            event_info['First response'] = [loaded_bpod_file['SessionData']['FirstPoke'][trial]]

            correct_side = loaded_bpod_file['SessionData']['TrialSide'][trial]
            if correct_side == 1:
                correct_port_in = 'Port1In'
                correct_port_out = 'Port1Out'
                reward_time = original_raw_events[trial]['States']['LeftReward'][0]
            else:
                correct_port_in = 'Port3In'
                correct_port_out = 'Port3Out'
                reward_time = original_raw_events[trial]['States']['RightReward'][0]
            all_correct_pokes_in = np.squeeze(np.asarray([original_raw_events[trial]['Events'][correct_port_in]]))

            if all_correct_pokes_in.size == 1 and all_correct_pokes_in >= reward_time:
                event_info['Time start'] = all_correct_pokes_in
            elif all_correct_pokes_in.size > 1:
                event_info['Time start'] = all_correct_pokes_in[
                    np.squeeze(np.where(all_correct_pokes_in > reward_time)[0])]
                if (event_info['Time start']).size > 1:
                    event_info['Time start'] = event_info['Time start'][0]
            else:
                event_info['Time start'] = np.empty(0)

            if trial < original_state_timestamps_all_trials.shape[0] - 1:
                if correct_port_out in original_raw_events[trial + 1]['Events']:
                    all_correct_pokes_out = np.squeeze(np.asarray([original_raw_events[trial + 1]['Events'][correct_port_out]]))
                    if event_info['Time start'].size != 0:
                        if all_correct_pokes_out.size == 1:
                            event_info['Time end'] = all_correct_pokes_out
                        elif (all_correct_pokes_out).size > 1:
                            indices = np.where(all_correct_pokes_out > 0)
                            if len(indices) > 1:
                                event_info['Time end'] = all_correct_pokes_out[0]
                            else:
                                event_info['Time end'] = all_correct_pokes_out[0]
                        else:
                            event_info['Time end'] = np.empty(0)

                        if (event_info['Time end']).size > 1:
                            event_info['Time end'] = event_info['Time end'][0]

                        event_info['Time end'] = [event_info['Time end'] + daq_trials_start_ttls[trial + 1]]
                    else:
                        event_info['Time end'] = [np.empty(0)]
                else:
                    event_info['Time end'] = [np.empty(0)]
            else: event_info['Time end'] = [np.empty(0)]
            event_info['Time start'] = [event_info['Time start'] + daq_trials_start_ttls[trial]]
        else:
            state_info['First choice correct'] = np.zeros(num_states)
            event_info['Trial num'] = [trial]
            event_info['Trial type'] = [loaded_bpod_file['SessionData']['TrialSequence'][trial]]
            event_info['State type'] = [5.5]
            event_info['Instance in state'] = [1]
            event_info['Max times in state'] = [1]
            if hasattr(loaded_bpod_file['SessionData']['TrialSettings'][trial], 'RewardChangeBlock'):
                event_info['Reward block'] = loaded_bpod_file['SessionData']['TrialSettings'][trial].RewardChangeBlock
            event_info['State name'] = ['First incorrect choice']
            out_of_centre_time = original_raw_events[trial]['States']['WaitForResponse'][0]
            correct_side = loaded_bpod_file['SessionData']['TrialSide'][trial]
            if correct_side == 1:
                incorrect_port_in = 'Port3In'
                incorrect_port_out = 'Port3Out'
            else:
                incorrect_port_in = 'Port1In'
                incorrect_port_out = 'Port1Out'

            if incorrect_port_in in original_raw_events[trial]['Events'] and incorrect_port_out in original_raw_events[trial]['Events']:
                all_incorrect_pokes_in = np.squeeze(np.asarray([original_raw_events[trial]['Events'][incorrect_port_in]]))
                all_incorrect_pokes_out = np.squeeze(np.asarray([original_raw_events[trial]['Events'][incorrect_port_out]]))
                if all_incorrect_pokes_in.size == 1 and all_incorrect_pokes_in > out_of_centre_time:
                    event_info['Time start'] = all_incorrect_pokes_in
                elif all_incorrect_pokes_in.size > 1:
                    event_info['Time start'] = all_incorrect_pokes_in[np.squeeze(np.where(all_incorrect_pokes_in > out_of_centre_time)[0])]
                    if (event_info['Time start']).size > 1:
                        event_info['Time start'] = event_info['Time start'][0]
                else: event_info['Time start'] = np.empty(0)



                if event_info['Time start'].size != 0:
                    if all_incorrect_pokes_out.size == 1 and all_incorrect_pokes_out > event_info['Time start']:
                        event_info['Time end'] = all_incorrect_pokes_out
                    elif (all_incorrect_pokes_out).size > 1:
                        indices = np.where(all_incorrect_pokes_out > event_info['Time start'])
                        if len(indices) > 1:
                            event_info['Time end'] = all_incorrect_pokes_out[np.squeeze(np.where(all_incorrect_pokes_out > event_info['Time start'])[0])]
                        else:
                            event_info['Time end'] = all_incorrect_pokes_out[
                                np.squeeze(np.where(all_incorrect_pokes_out > event_info['Time start']))]
                    else: event_info['Time end'] = np.empty(0)

                    if (event_info['Time end']).size > 1:
                        event_info['Time end'] = event_info['Time end'][0]

                    event_info['Time end'] = [event_info['Time end'] + daq_trials_start_ttls[trial]]

                else: event_info['Time end'] = []
                event_info['Time start'] = [event_info['Time start'] + daq_trials_start_ttls[trial]]

                event_info['Response'] = [loaded_bpod_file['SessionData']['ChosenSide'][trial]]
                if trial > 0:
                    event_info['Last response'] = [loaded_bpod_file['SessionData']['ChosenSide'][
                        trial - 1]]
                    event_info['Last outcome'] = [loaded_bpod_file['SessionData']['Outcomes'][
                        trial - 1]]
                else:
                    event_info['Last response'] = [-1]
                    event_info['Last outcome'] = [-1]
                event_info['Trial start'] = [daq_trials_start_ttls[trial]]
                event_info['Trial end'] = [(state_timestamps[-1] + daq_trials_start_ttls[trial])]
                event_info['Trial outcome'] = [loaded_bpod_file['SessionData']['Outcomes'][trial]]
                event_info['First response'] = [loaded_bpod_file['SessionData']['FirstPoke'][trial]]
            else: event_info = {}

        trial_data = pd.DataFrame(state_info)

        if trial == 0:
            restructured_data = trial_data
        else:
            restructured_data = pd.concat([restructured_data, trial_data], ignore_index=True)
        if event_info != {} and event_info['Time start'][0].size != 0 and event_info['Time end'][0].size != 0:
            event_data = pd.DataFrame(event_info)
            restructured_data = pd.concat([restructured_data, event_data], ignore_index=True)
    return restructured_data


def get_poke_times(loaded_bpod_file, trial_start_ttls_daq, clock_pulses):
    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
    original_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

    daq_trials_start_ttls = trial_start_ttls_daq
    all_port1_pokes_in = np.array([])
    all_port2_pokes_out = np.array([])
    all_port3_pokes_in = np.array([])
    # loops through all the trials and pulls out all the states
    for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):
        trial_start = daq_trials_start_ttls[trial]
        if 'Port1In' in original_raw_events[trial]['Events']:
            port1_pokes_in = np.squeeze(np.asarray([original_raw_events[trial]['Events']['Port1In']])) + trial_start
            all_port1_pokes_in = np.append(all_port1_pokes_in, port1_pokes_in)
        if 'Port3In' in original_raw_events[trial]['Events']:
            port3_pokes_in = np.squeeze(np.asarray([original_raw_events[trial]['Events']['Port3In']])) + trial_start
            all_port3_pokes_in = np.append(all_port3_pokes_in, port3_pokes_in)
        if 'Port2Out' in original_raw_events[trial]['Events']:
            port2_pokes_out = np.squeeze(np.asarray([original_raw_events[trial]['Events']['Port2Out']])) + trial_start
            all_port2_pokes_out = np.append(all_port2_pokes_out, port2_pokes_out)

    return all_port1_pokes_in, all_port2_pokes_out, all_port3_pokes_in

