import sys
import os
import pandas as pd
import nptdms
import numpy as np
import math
import scipy.io
import scipy.signal

sys.path.insert(1, 'C:\\Users\\Yvonne\\Documents\\freely_moving_photometry_analysis')

def get_all_experimental_records():
    experiment_record = pd.read_csv('C:\\Users\\Yvonne\\Documents\\experimental_record.csv')
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record

def pre_process_experiments(experiments, method='pyphotometry', protocol='Two_Alternative_Choice', folder_path='Z:\\users\\Yvonne\\photometry_2AC\\', saving_folder_path='Z:\\users\\Yvonne\\photometry_2AC\\'):
    for index, experiment in experiments.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        pre_process_experiment_lerner_deissroth(mouse, date, protocol, folder_path, saving_folder_path)
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(temp_dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in temp_dict:
        if isinstance(temp_dict[key], scipy.io.matlab.mio5_params.mat_struct):
            temp_dict[key] = _todict(temp_dict[key])

    return temp_dict
def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    temp_dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            temp_dict[strg] = _todict(elem)
        else:
            temp_dict[strg] = elem
    return temp_dict

def pre_process_experiment_lerner_deissroth(mouse, date, protocol, folder_path, saving_folder_path):
    daq_file = find_daq_file(mouse, date, folder_path, protocol)
    data = nptdms.TdmsFile(daq_file)
    sampling_rate = 10000

    main_session_file = find_bpod_file(mouse, date, protocol, folder_path, saving_folder_path)
    loaded_bpod_file, trial_raw_events = load_bpod_file(main_session_file)

    chan_0 = data.group_channels('acq_task')[0].data
    led405 = data.group_channels('acq_task')[2].data
    led465 = data.group_channels('acq_task')[1].data
    clock = data.group_channels('acq_task')[3].data
    stim_trigger = data.group_channels('acq_task')[4].data
    stim_trigger_gaps = np.diff(stim_trigger)
    trial_start_ttls_daq_samples = np.where(stim_trigger_gaps > 2.6)
    trial_start_ttls_daq = trial_start_ttls_daq_samples[0] / sampling_rate
    daq_num_trials = trial_start_ttls_daq.shape[0]
    bpod_num_trials = trial_raw_events.shape[0]
    if daq_num_trials != bpod_num_trials:
        print('numbers of trials do not match! lerner_deisseroth_preprocess')
        print('daq: ', daq_num_trials)
        print('bpod: ', bpod_num_trials)

    else:
        print(daq_num_trials, 'trials in session (lerner_deisseroth_preprocess)')

    df_clipped = lerner_deisseroth_preprocess(chan_0[sampling_rate * 6:], led465[sampling_rate * 6:],
                                              led405[sampling_rate * 6:], sampling_rate)
    df = np.pad(df_clipped, (6 * sampling_rate, 0), mode='median')
    clock_diff = np.diff(clock)
    clock_pulses = np.where(clock_diff > 2.6)[0] / sampling_rate

    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
    daq_trials_start_ttls = trial_start_ttls_daq


    if protocol is 'Random_Tone_Clouds':
        restructured_data = restructure_bpod_timestamps_random_tone_clouds(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)
    elif protocol is 'Random_WN':
        restructured_data = restructure_bpod_timestamps_random_tone_clouds(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)
    elif protocol is 'AirpuffPhoto':
        restructured_data = restructure_bpod_timestamps_airpuff(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)
    else: # standard analysis
        restructured_data = restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)

    saving_folder = folder_path + 'processed_data\\' + mouse + '\\'


    if protocol is 'Random_Tone_Clouds':
        smoothed_trace_filename = mouse + '_' + date + '_RTC_smoothed_signal.npy'
    elif protocol is 'Random_WN':
        smoothed_trace_filename = mouse + '_' + date + '_RWN_smoothed_signal.npy'
    elif protocol is 'AirpuffPhoto':
        smoothed_trace_filename = mouse + '_' + date + '_Airpuff_smoothed_signal.npy'
    else:
        smoothed_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'


    if protocol is 'Random_Tone_Clouds':
        restructured_data_filename = mouse + '_' + date + '_RTC_restructured_data.pkl'
    elif protocol is 'Random_WN':
        restructured_data_filename = mouse + '_' + date + '_RWN_restructured_data.pkl'
    elif protocol is 'AirpuffPhoto':
        restructured_data_filename = mouse + '_' + date + '_Airpuff_restructured_data.pkl'
    else:
        restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'


    np.save(saving_folder + smoothed_trace_filename, df)
    restructured_data.to_pickle(saving_folder + restructured_data_filename)

def find_daq_file(mouse, date, folder_path, protocol):

    #daq_data_path = 'Z:\\users\\Yvonne\\photometry_2AC\\freely_moving_photometry_data\\' + mouse + '\\' #+ mouse + '_RTC' + '\\' #'W:\\photometry_2AC\\freely_moving_photometry_data\\' + mouse + '\\'

    if protocol is 'Random_Tone_Clouds':
        daq_data_path = folder_path + 'freely_moving_photometry_data\\' + mouse + '\\' + mouse + '_RTC' + '\\'
    elif protocol is 'Random_WN':
        daq_data_path = folder_path + 'freely_moving_photometry_data\\' + mouse + '\\' + mouse + '_RWN' + '\\'
    elif protocol is 'AirpuffPhoto':
        daq_data_path = folder_path + 'freely_moving_photometry_data\\' + mouse + '\\' + mouse + '_airpuff' + '\\'
    else:
        daq_data_path = folder_path + 'freely_moving_photometry_data\\' + mouse + '\\'
    #daq_data_path ='C:\\Users\\francescag\\Documents\\PhD_Project\\SNL_photo_photometry\\freely_moving_photometry_data\\' + mouse + '\\'
    folders_in_photo_path = os.listdir(daq_data_path)

    folders_on_that_day = [s for s in folders_in_photo_path if date in s]

    if len(folders_on_that_day) == 2:
        main_session_file = daq_data_path + '\\' + folders_on_that_day[-1] + '\\' + 'AI.tdms'
        return main_session_file
        print('2 sessions that day')
    elif len(folders_on_that_day) == 1:
        main_session_file = daq_data_path + '\\' + folders_on_that_day[-1] + '\\' + 'AI.tdms'
        return main_session_file
    elif len(folders_on_that_day) == 0:
        print('0 sessions found!')
    else:
        print('More than 2 sessions that day!')


def find_bpod_file(mouse, date, protocol, folder_path, saving_folder_path):
    Bpod_data_path = folder_path + 'bpod_data\\' + mouse + '\\' + protocol + '\\Session Data\\'
    #Bpod_data_path = 'V:\\users\\Yvonne\\photometry_2AC\\bpod_data\\' + mouse + '\\' + protocol_type + '\\Session Data\\' #'W:\\photometry_2AC\\bpod_data\\' + mouse + '\\' + protocol_type + '\\Session Data\\'
    #Bpod_data_path = 'C:\\Users\\francescag\\Documents\\PhD_Project\\SNL_photo_photometry\\bpod_data\\'+ mouse + '\\' + protocol_type + '\\Session Data\\'
    Bpod_file_search_tool = mouse + '_' + protocol + '_' + date
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
    elif len(mat_files_on_that_day) == 0:
        print('0 sessions that day!')
    else:
        print('More than 2 sessions that day!')


def load_bpod_file(main_session_file):
    # gets the Bpod data out of MATLAB struct and into python-friendly format
    #print(main_session_file)
    loaded_bpod_file = loadmat(main_session_file)
    #loaded_bpod_file = load_ns.loadmat(main_session_file)
    #loaded_bpod_file = bpod.load_bpod_file(main_session_file)

    # as RawEvents.Trial is a cell array of structs in MATLAB, we have to loop through the array and convert the structs to dicts
    trial_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

    for trial_num, trial in enumerate(trial_raw_events):
        trial_raw_events[trial_num] = _todict(trial)
        #trial_raw_events[trial_num] = load_ns._todict(trial)

    loaded_bpod_file['SessionData']['RawEvents']['Trial'] = trial_raw_events
    trial_settings = loaded_bpod_file['SessionData']['TrialSettings']
    first_trial = trial_raw_events[0]
    first_trial_TTL = first_trial['States']['TrialStart']
    return loaded_bpod_file, trial_raw_events



def lerner_deisseroth_preprocess(
    photodetector_raw_data,
    reference_channel_211hz,
    reference_channel_531hz,
    sampling_rate,
):
    """
    process data according to https://www.ncbi.nlm.nih.gov/pubmed/26232229 , supplement 11
    :param photodetector_raw_data: the raw signal from the photodetector
    :param reference_channel_211hz:  a copy of the reference signal sent to the signal LED (Ca2+ dependent)
    :param reference_channel_531hz:  a copy of the reference signal sent to the background LED (Ca2+ independent)
    :return: deltaF / F
    """
    demodulated_211, demodulated_531 = demodulate(
        photodetector_raw_data,
        reference_channel_211hz,
        reference_channel_531hz,
        sampling_rate,
    )

    signal = _apply_butterworth_lowpass_filter(
        demodulated_211, 2, sampling_rate, order=2
    )
    background = _apply_butterworth_lowpass_filter(
        demodulated_531, 2, sampling_rate, order=2
    )

    regression_params = np.polyfit(background, signal, 1)
    bg_fit = regression_params[0] * background + regression_params[1]

    delta_f = (signal - bg_fit) / bg_fit
    return delta_f

def demodulate(raw, ref_211, ref_531, sampling_rate):
    """
    gets demodulated signals for 211hz and 531hz am modulated signal
    :param raw:
    :param ref_211:
    :param ref_531:
    :return:
    """

    q211, i211 = am_demodulate(raw, ref_211, 211, sampling_rate=sampling_rate)
    q531, i531 = am_demodulate(raw, ref_531, 531, sampling_rate=sampling_rate)
    demodulated_211 = _demodulate_quadrature(q211, i211)
    demodulated_531 = _demodulate_quadrature(q531, i531)

    return demodulated_211, demodulated_531

def _demodulate_quadrature(quadrature, in_phase):
    return (quadrature ** 2 + in_phase ** 2) ** 0.5


def _apply_butterworth_lowpass_filter(
    demod_signal, low_cut_off=15, fs=10000, order=5):
    w = low_cut_off / (fs / 2)  # Normalize the frequency
    b, a = scipy.signal.butter(order, w, "low")
    output = scipy.signal.filtfilt(b, a, demod_signal)
    return output


def am_demodulate(signal, reference, modulation_frequency, sampling_rate=10000, low_cut=15, order=5):
    normalised_reference = reference - reference.mean()
    samples_per_period = sampling_rate / modulation_frequency
    samples_per_quarter_period = round(samples_per_period / 4)

    shift_90_degrees = np.roll(normalised_reference, samples_per_quarter_period)
    in_phase = np.pad(signal * normalised_reference, (sampling_rate, 0), mode='median')
    in_phase_filtered_pad = _apply_butterworth_lowpass_filter(in_phase, low_cut_off=low_cut, fs=sampling_rate,
                                                              order=order)
    in_phase_filtered = in_phase_filtered_pad[sampling_rate:]

    quadrature = np.pad(signal * shift_90_degrees, (sampling_rate, 0), mode='median')
    quadrature_filtered_pad = _apply_butterworth_lowpass_filter(quadrature, low_cut_off=low_cut, fs=sampling_rate,
                                                                order=order)
    quadrature_filtered = quadrature_filtered_pad[sampling_rate:]

    return quadrature_filtered, in_phase_filtered


def restructure_bpod_timestamps_random_tone_clouds(loaded_bpod_file, trial_start_ttls_daq, clock_pulses):
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
        sound_types = {'COT': 0, 'NA': 1, 'WN':2}
         # need to silence sound type YJ for analysing TS5, 230921; FG didn't use sound types in the past
        state_info['Sound type'] = np.ones((num_states)) * sound_types[loaded_bpod_file['SessionData']['SoundType'][trial]]
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
        #state_info['Start camera frame'] = find_nearest(clock_pulses, state_info['Time start'])
        #state_info['End camera frame'] = find_nearest(clock_pulses, state_info['Time end'])

        state_info['Trial start'] = np.ones((num_states)) * daq_trials_start_ttls[trial]
        state_info['Trial end'] = np.ones((num_states)) * (state_timestamps[-1] + daq_trials_start_ttls[trial])
        trial_data = pd.DataFrame(state_info)

        if trial == 0:
            restructured_data = trial_data
        else:
            restructured_data = pd.concat([restructured_data, trial_data], ignore_index=True)
        if event_info != {} and event_info['Time start'][0].size != 0 and event_info['Time end'][0].size != 0:
            event_data = pd.DataFrame(event_info)
            restructured_data = pd.concat([restructured_data, event_data], ignore_index=True)
    return restructured_data


def restructure_bpod_timestamps_airpuff(loaded_bpod_file, trial_start_ttls_daq, clock_pulses):
    trial_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']

    for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):
        state_info = {}
        trial_states = original_state_data_all_trials[trial]
        num_states = (len(trial_states))
        state_info['Trial num'] = np.ones((num_states)) * trial
        state_info['State type'] = original_state_data_all_trials[trial]
        state_info['State name'] = list(trial_raw_events[trial]['States'])          # pycharm functional version
        #state_info['State name'] = list(_todict(trial_raw_events[trial].States))   # this is the only way to make it work in jupyter!
        state_info['Trial start'] = np.ones((num_states)) * trial_start_ttls_daq[trial]
        state_info['Trial end'] = np.ones((num_states)) * (state_timestamps[-1] + trial_start_ttls_daq[trial])
        state_info['Time start'] = state_timestamps[0:-1] + trial_start_ttls_daq[trial]
        state_info['Time end'] = state_timestamps[1:] + trial_start_ttls_daq[trial]
        state_info['Duration'] = state_info['Time end'] - state_info['Time start']

        trial_data = pd.DataFrame(state_info)

        if trial == 0:
            restructured_data = trial_data
        else:
            restructured_data = pd.concat([restructured_data, trial_data], ignore_index=True)
    return restructured_data


def restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq, clock_pulses):
    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps'] #an array of arrays, each array consists of the timestamps of a trial
    original_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

    daq_trials_start_ttls = trial_start_ttls_daq # an array of the actual timestamps of the trial start TTLs
    # loops through all the trials and pulls out all the states
    for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):
        # now trial as in pycharm (starts with 0), not as in matlab (starts with counting at 1):
        state_info = {} # a dict containing trial by trial all info; each trial has the length of the number of states in that trial
        event_info = {}
        trial_states = original_state_data_all_trials[trial]
        num_states = (len(trial_states))
        sound_types = {'COT': 0, 'NA': 1} # 2AC classic: all COT, all 0; SOR: COT = 0, NA = 1 = SOR; Silence:
        try:
            state_info['Sound type'] = np.ones((num_states)) * sound_types[loaded_bpod_file['SessionData']['SoundType'][trial]] # 0 = COT, 1 = NA
            # matlab sound type is a string of COT or NA;
        except:
            state_info['Sound type'] = np.ones((num_states)) * 10

        state_info['Trial num'] = np.ones((num_states)) * trial
        state_info['Trial type'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['TrialSequence'][trial] # 1 or 7 depending on high or low frequency sound
        state_info['State type'] = trial_states # all states in that trial
        num_times_in_state = find_num_times_in_state(trial_states)
        state_info['Instance in state'] = num_times_in_state[0]     # 1st, 2nd, 3rd time state X is happening
        state_info['Max times in state'] = num_times_in_state[1]    # max N that state X has happened
        state_info['State name'] = loaded_bpod_file['SessionData']['RawData']['OriginalStateNamesByNumber'][0][
            trial_states - 1] # -1 because MATLAB is 1-indexed while python is 0-indexed
        state_info['Time start'] = state_timestamps[0:-1] + daq_trials_start_ttls[trial]    # all but the last time stamp + the trial start TTL
        state_info['Time end'] = state_timestamps[1:] + daq_trials_start_ttls[trial]        # all but the first time stamp + the trial start TTL
        #state_info['Start camera frame'] = find_nearest(clock_pulses, state_info['Time start'])
        #state_info['End camera frame'] = find_nearest(clock_pulses, state_info['Time end'])
        state_info['Response'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['ChosenSide'][trial] # 2 = right, 1 = left

        if trial > 0:
            state_info['Last response'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['ChosenSide'][
                trial - 1]
            state_info['Last outcome'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['Outcomes'][trial - 1]
        else:
            state_info['Last response'] = np.ones((num_states)) * -1
            state_info['Last outcome'] = np.ones((num_states)) * -1

        state_info['Trial start'] = np.ones((num_states)) * daq_trials_start_ttls[trial] # trial start TTL
        state_info['Trial end'] = np.ones((num_states)) * (state_timestamps[-1] + daq_trials_start_ttls[trial]) # last state timestamp + trial start TTL
        state_info['Trial outcome'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['Outcomes'][trial] # 1 = correct, 3 = in SOR: didn't poke in time
        state_info['First response'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['FirstPoke'][trial]
        if hasattr(loaded_bpod_file['SessionData']['TrialSettings'][trial], 'RewardChangeBlock'):
            state_info['Reward block'] = np.ones((num_states)) * loaded_bpod_file['SessionData']['TrialSettings'][trial].RewardChangeBlock

        if loaded_bpod_file['SessionData']['FirstPoke'][trial] == loaded_bpod_file['SessionData']['TrialSide'][trial]: # correct
            state_info['First choice correct'] = np.ones(num_states)
            event_info['First choice correct'] = [1]
            event_info['Trial num'] = [trial]
            try:
                event_info['Sound type'] = sound_types[loaded_bpod_file['SessionData']['SoundType'][trial]]
            except KeyError:
                event_info['Sound type'] = 10

            event_info['Trial type'] = [loaded_bpod_file['SessionData']['TrialSequence'][trial]]
            event_info['State type'] = [8.5]    # correct choice, State = leaving reward port
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

            correct_side = loaded_bpod_file['SessionData']['TrialSide'][trial] # 1 = left was correct, 2 = right was correct
            if correct_side == 1: # mouse should have gone left
                correct_port_in = 'Port1In' # left port
                correct_port_out = 'Port1Out' # left port
                reward_time = original_raw_events[trial]['States']['LeftReward'][0]
            else:
                correct_port_in = 'Port3In'
                correct_port_out = 'Port3Out'
                reward_time = original_raw_events[trial]['States']['RightReward'][0]
            all_correct_pokes_in = np.squeeze(np.asarray([original_raw_events[trial]['Events'][correct_port_in]]))
            #print('all_correct_pokes_in.size: ' + str(all_correct_pokes_in.size) + ', reward_time: ' + str(reward_time) + ', trial: ' + str(trial))
            if all_correct_pokes_in.size == 1 and all_correct_pokes_in >= reward_time:
                #print('        1 all_correct_pokes_in: ' + str(all_correct_pokes_in))
                event_info['Time start'] = all_correct_pokes_in
            elif all_correct_pokes_in.size > 1:
                event_info['Time start'] = all_correct_pokes_in[
                    np.squeeze(np.where(all_correct_pokes_in > reward_time)[0])]
                if (event_info['Time start']).size > 1:
                    event_info['Time start'] = event_info['Time start'][0]
            else:
                event_info['Time start'] = np.empty(0)

            if trial < original_state_timestamps_all_trials.shape[0] - 1: # if not the last trial
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
                        else: # all_correct_pokes_out.size = 0
                            event_info['Time end'] = np.empty(0)

                        if (event_info['Time end']).size > 1:
                            event_info['Time end'] = event_info['Time end'][0]

                        event_info['Time end'] = [event_info['Time end'] + daq_trials_start_ttls[trial + 1]]
                    else: # if there was not start time for correct poke in
                        event_info['Time end'] = [np.empty(0)]
                else: # if the next trial doesn't have a correct_port_out
                    event_info['Time end'] = [np.empty(0)]
            else: # if it is the last trial
                event_info['Time end'] = [np.empty(0)]
            event_info['Time start'] = [event_info['Time start'] + daq_trials_start_ttls[trial]]


        else: # incorrect trial or a missed trial
            out_of_centre_time = original_raw_events[trial]['States']['WaitForResponse'][0] # MOVED UP YJ: FIRST CHECK IF THIS IS AN INCORRECT TRIAL OR A MISSED (TIME OUT) TRIAL
            if math.isnan(out_of_centre_time) == False:                  # ADDED BY YJ

            # added 1 indent YJ from here

                state_info['First choice correct'] = np.zeros(num_states) # 0 for incorrect
                event_info['Trial num'] = [trial]
                event_info['Trial type'] = [loaded_bpod_file['SessionData']['TrialSequence'][trial]] # 1 or 7, high or low f
                event_info['State type'] = [5.5] # wrong choice, first incorrect choice
                try:
                    event_info['Sound type'] = sound_types[
                        loaded_bpod_file['SessionData']['SoundType'][trial]] # COT or NA
                except KeyError:
                    event_info['Sound type'] = 'not available'


                event_info['Instance in state'] = [1]
                event_info['Max times in state'] = [1]
                if hasattr(loaded_bpod_file['SessionData']['TrialSettings'][trial], 'RewardChangeBlock'):
                    event_info['Reward block'] = loaded_bpod_file['SessionData']['TrialSettings'][trial].RewardChangeBlock
                event_info['State name'] = ['First incorrect choice']
                #out_of_centre_time = original_raw_events[trial]['States']['WaitForResponse'][0] # Time start of Wait for response              # SILENCED YJ
                    # trouble when this is nan,e.g. in SOR because 5s passed before WaitForResponse state occured (WaitforResp = nan)
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
                    if all_incorrect_pokes_in.size == 1 and all_incorrect_pokes_in > out_of_centre_time: # an incorrect poke after wait for response has started
                        event_info['Time start'] = all_incorrect_pokes_in
                    elif all_incorrect_pokes_in.size > 1:
                        #print('trial: ' + str(trial) + ' first choice incorrect, correct side = ' + str(correct_side) + '; all_incorrect_pokes_in: ' + str(all_incorrect_pokes_in) + ' out_of_centre_time: ' + str(out_of_centre_time))
                        #print('event_info[Time start]: ' + str(event_info['Time start']))
                        #print('out_of_centre_time: ' + str(out_of_centre_time))
                        #print('----------------')
                        event_info['Time start'] = all_incorrect_pokes_in[np.squeeze(np.where(all_incorrect_pokes_in > out_of_centre_time)[0])]
                        if (event_info['Time start']).size > 1:
                            event_info['Time start'] = event_info['Time start'][0]
                    else:
                        event_info['Time start'] = np.empty(0)

                    if event_info['Time start'].size != 0:
                        if all_incorrect_pokes_out.size == 1 and all_incorrect_pokes_out > event_info['Time start']:
                            event_info['Time end'] = all_incorrect_pokes_out
                        elif all_incorrect_pokes_out.size > 1:
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
                else:
                    event_info = {}
        # analysis of incorrect trials end

        trial_data = pd.DataFrame(state_info)

        # no one extra indent
        if trial == 0:
            restructured_data = trial_data
        else:
            restructured_data = pd.concat([restructured_data, trial_data], ignore_index=True)
            if event_info != {} and event_info['Time start'][0].size != 0 and event_info['Time end'][0].size != 0:
                event_data = pd.DataFrame(event_info)
                restructured_data = pd.concat([restructured_data, event_data], ignore_index=True)
    return restructured_data

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

