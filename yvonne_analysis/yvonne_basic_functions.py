import os.path
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import peakutils


def get_all_experimental_records():
    experiment_record = pd.read_csv('C:\\Users\\Yvonne\\Documents\\experimental_record.csv')
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record

def analyse_this_experiment(experiment_to_process, main_dir):
    for index, experiment in experiment_to_process.iterrows():
        saving_folder = main_dir + 'YJ_results\\' + experiment['mouse_id']
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        try:
            session_traces = SessionData(experiment['mouse_id'], experiment['date'], experiment['fiber_side'], experiment['recording_site'], main_dir)
            filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + experiment['fiber_side']+ '_' + experiment['recording_site'] + '_aligned_traces.p'
            save_filename = os.path.join(saving_folder, filename)
            pickle.dump(session_traces, open(save_filename, "wb"))
            return session_traces
        except OSError:
            print('No preprocessed files found for: ' + experiment['mouse_id'] + '_' + experiment['date'] + '_' + experiment['fiber_side']+ '_' + experiment['recording_site'])
def get_SessionData(directory, mouse, date, fiber_side, location):
    folder = directory + 'YJ_results\\' + mouse + '\\'
    aligned_filename = folder + mouse + '_' + date + '_' + fiber_side + '_' + location + '_' + 'aligned_traces.p'
    with open(aligned_filename, 'rb') as f:
        data = pickle.load(f)
    return data

def fiber_side_to_numeric(fiber_side):
    #fiber_options = np.array(['left', 'right'])#left = 1, right = 2
    if(fiber_side == 'left'):
        ipsi_numeric = '1'
        contra_numeric = '2'
    elif(fiber_side == 'right'):
        ipsi_numeric = '2'
        contra_numeric = '1'
    return ipsi_numeric, contra_numeric


def getProtocol(session_data):
    processed_folder = session_data.directory + 'processed_data\\' + session_data.mouse + '\\'
    restructured_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(processed_folder + restructured_filename)
    protocol = ''
    total_trials = np.max(trial_data['Trial num']) + 1  # trial 0 is the first trial
    COT_types = len(trial_data['Trial type'].unique())
    if COT_types > 2:
        if COT_types == 7:
            protocol = protocol + 'psychometric; '
    sound_types = len(trial_data['Sound type'].unique())
    if sound_types > 1:
        silent_events = trial_data[trial_data['Sound type'] == 1]
        sound_events = trial_data[trial_data['Sound type'] == 0]
        silent_trials = len(silent_events['Trial num'].unique())
        sound_trials = sound_events['Trial num'].unique()
        if silent_trials > 0:
            protocol = protocol + 'silence (' + str("%.1f" % (silent_trials / total_trials * 100)) + '%); '

    if protocol == '':
        protocol = '2AC'
    return protocol


def getPerformance(session_data):
    processed_folder = session_data.directory + 'processed_data\\' + session_data.mouse + '\\'
    restructured_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
    try:
        trial_data = pd.read_pickle(processed_folder + restructured_filename)
        total_trials = np.max(trial_data['Trial num'])+1 #trial 0 is the first trial
        punishment = trial_data[trial_data['State name']=='Punish']
        if len(punishment) > 0:
            wrong_trials = len(punishment['Trial num'].unique())
            correct_trials = total_trials - wrong_trials
            performance = correct_trials / total_trials * 100
        else:
            correct_trial_data = trial_data[trial_data['First choice correct']==1]
            correct_trials = len(correct_trial_data['Trial num'].unique())
            performance = correct_trials / total_trials * 100
        return performance, total_trials
    except OSError:
        print('No trial data found for ' + session_data.mouse + '_' + session_data.date + '_' + session_data.fiber_side + '_' + session_data.recording_site)


class SessionData(object):
    def __init__(self, mouse_id, date, fiber_side, recording_site, main_dir):
        self.mouse = mouse_id
        self.date = date
        self.fiber_side = fiber_side
        self.recording_site = recording_site
        self.directory = main_dir
        self.performance, self.nr_trials = getPerformance(self)
        self.protocol = getProtocol(self)
        self.choice = None
        self.cue = None
        self.reward = None
        self.outcome = None
        self.return_data = None


        print('Processing session: ' + self.mouse + '_' + self.date + '_' + self.fiber_side + '_' + self.recording_site + '_movement / choice:')
        self.choice = ChoiceAlignedData(self, save_traces=True)
        print('Processing session: ' + self.mouse + '_' + self.date + '_' + self.fiber_side + '_' + self.recording_site + '_cue:')
        self.cue = CueAlignedData(self, save_traces=True)
        print('Processing session: ' + self.mouse + '_' + self.date + '_' + self.fiber_side + '_' + self.recording_site + '_reward:')
        self.reward = RewardAlignedData(self, save_traces=True)
        #self.outcome_data = RewardAndNoRewardAlignedData(self, save_traces=save_traces)




class ChoiceAlignedData(object):
    """
    Traces for standard analysis
    """
    def __init__(self, session_data, save_traces = True):
        processed_folder = session_data.directory + 'processed_data\\' + session_data.mouse + '\\'
        restructured_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(processed_folder + restructured_filename)
        dff_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(processed_folder + dff_filename)

        # "RESPONSE": RIGHT = 2, LEFT = 1: hence ipsi and contra need to be assigned accordingly:
        fiber_options = np.array(['left', 'right'])     # left = (0+1) = 1; right = (1+1) == 2
        ipsi_fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0]+1)[0]      # if fiber on right ipsi = 2; if fiber on left ipsi  = 1
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0]+1)[0]    # if fiber on right contra = 1, if fiber on left contra = 2

        params = {'state_type_of_interest': 5,
            'outcome': 2, # doesn't matter for choice aligned data
            # 'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0, # doesnt matter for choice aligned data
            'align_to' : 'Time start',
            'instance': -1, # last instance
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, ipsi_fiber_side_numeric, ipsi_fiber_side_numeric)
        self.ipsi_data.get_peaks(save_traces=save_traces)

        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.contra_data.get_peaks(save_traces=save_traces)


class CueAlignedData(object):
    def __init__(self, session_data, save_traces=True):
        processed_folder = session_data.directory + 'processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(processed_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(processed_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        ipsi_fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 3,
            'outcome': 2,
            #'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 0, # the cue is often repeated
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, ipsi_fiber_side_numeric, ipsi_fiber_side_numeric)
        self.ipsi_data.get_peaks(save_traces=save_traces)
        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.contra_data.get_peaks(save_traces=save_traces)

        #params['cue'] = 'high'
        #self.high_cue_data = ZScoredTraces(trial_data, dff, params, 0, 0) # no matter if ipsi or contra, high cue is always 0
        #self.high_cue_data.get_peaks(save_traces=save_traces)
        #self.high_cue_ipsi_data = ZScoredTraces(trial_data, dff, params, ipsi_fiber_side_numeric, ipsi_fiber_side_numeric)
        #self.high_cue_contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        #params['cue'] = 'low'
        #self.low_cue_data = ZScoredTraces(trial_data, dff, params, 0, 0)
        #self.low_cue_data.get_peaks(save_traces=save_traces)
        #elf.low_cue_ipsi_data = ZScoredTraces(trial_data, dff, params, ipsi_fiber_side_numeric,
         #                                       ipsi_fiber_side_numeric)
        #self.low_cue_contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric,
        #                                          contra_fiber_side_numeric)

class RewardAlignedData(object):
    def __init__(self, session_data, save_traces=True):
        processed_folder = session_data.directory + 'processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(processed_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(processed_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        ipsi_fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 5,
                  'outcome': 1,
                  #'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': 1,
                  'cue': 'None'}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, ipsi_fiber_side_numeric, ipsi_fiber_side_numeric)
        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)



class ZScoredTraces(object):
    def __init__(self, trial_data, dff, params, response, first_choice):
        self.trial_peaks = None
        self.params = HeatMapParams(params, response, first_choice) # 1) params, 2) response (= L vs R corresp. to ipsi vs contra), 3) first_choice == X?? first response?
        self.time_points, self.mean_trace, self.sorted_traces, self.reaction_times, self.state_name, self.title, self.sorted_next_poke, self.trial_nums, self.event_times, self.outcome_times = find_and_z_score_traces(
            trial_data, dff, self.params, sort=False)
    def get_peaks(self, save_traces=True):
        if self.params.align_to == 'Time start':
            other_time_point = self.outcome_times   #outcome time is the time of reward or punsihment
        else: # for reward or non reward aligned data
            other_time_point = self.sorted_next_poke #first poke in subsequent trial
        self.trial_peaks = get_peak_each_trial_no_nans(self.sorted_traces, self.time_points, other_time_point)
        if not save_traces:
            self.sorted_traces = None

def get_peak_each_trial(sorted_traces, time_points, sorted_other_events):
    all_trials_peaks = []
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0), np.less_equal(time_points, sorted_other_events[trial_num])))
            # returns the indices of the time points that are between 0 and the time of the next poke or outcome
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        #plt.plot(time_points[indices_to_integrate], trial_trace)
        #plt.plot(sorted_other_event, np.zeros(sorted_other_event.shape[0]),'o')

        # trial_trace = trial_trace # - trial_trace[0]s # FG, unnecessary line
        trial_peak_inds = peakutils.indexes(trial_trace.flatten('F'))
        if len(trial_peak_inds>1):
            trial_peak_inds = trial_peak_inds[0]
        trial_peaks = trial_trace.flatten('F')[trial_peak_inds]
        all_trials_peaks.append(trial_peaks)
        #plt.plot(trial_trace)
        #plt.scatter(trial_peak_inds, trial_peaks)
    flat_peaks = all_trials_peaks
    #plt.show()
    return flat_peaks


def get_peak_each_trial_no_nans(sorted_traces, time_points, sorted_other_events):
    all_trials_peaks = []
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0), np.less_equal(time_points, sorted_other_events[trial_num])))
        # returns the indices of the time points that are between 0 and the time of the next poke or outcome
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        # trial_trace = trial_trace # - trial_trace[0]s
        trial_peak_inds = peakutils.indexes(trial_trace.flatten('F')) #flatten: no smoothing, just re-arrange the array
        if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
            trial_peak_inds = trial_peak_inds[0]
            trial_peaks = trial_trace.flatten('F')[trial_peak_inds]
            # added YJ:
            #trace = trial_trace.flatten('F')
            #plt.plot(trial_trace)
            #plt.plot(trial_peak_inds, trace[trial_peak_inds], 'o')

        else: # if there is no peak_inds, just take the max of the trace
            # this is the case when the max is at the very beginning or the end of the trace
            trial_peak_inds = np.argmax(trial_trace)
            trial_peaks = np.max(trial_trace)
        all_trials_peaks.append(trial_peaks)
    flat_peaks = all_trials_peaks
    #plt.show()
    return flat_peaks



class HeatMapParams(object):
    def __init__(self, params, response, first_choice):
        self.state = params['state_type_of_interest']
        self.outcome = params['outcome']
        #self.last_outcome = params['last_outcome']
        self.response = response
        self.last_response = params['last_response']
        self.align_to = params['align_to']
        self.other_time_point = np.array(['Time start', 'Time end'])[np.where(np.array(['Time start', 'Time end']) != params['align_to'])]
        self.instance = params['instance']
        self.plot_range = params['plot_range']
        self.no_repeats = params['no_repeats']
        self.first_choice_correct = params['first_choice_correct']
        self.first_choice = first_choice
        self.cue = params['cue']


def get_next_centre_poke(trial_data, events_of_int, last_trial):
    '''
    This function returns the time of the first centre poke in the subsequent trial for each event of interest.

    last_trial is a boolean that is true if the last trial in the session is included in the events of interest
    '''
    next_centre_poke_times = np.zeros(events_of_int.shape[0])
    events_of_int = events_of_int.reset_index(drop=True)
    for i, event in events_of_int.iterrows(): # all but the last trial in events of interest
        trial_num = event['Trial num']
        if trial_num == trial_data['Trial num'].values[-1]: # if last trial in session; however iterrows does not include the last trial
            #if last_trial: #FG, if last trial in events of int = last trial in session, if is true for all trials
            next_centre_poke_times[i] = events_of_int['Trial end'].values[i] + 2
            #print('     fct get_next_centre_poke: last trial ' + str(trial_num) + ' time: ' + str(next_centre_poke_times[i]) + ' it did enter the if statement')
        else:   # YJ, it should go here for all trials but the last one! not the case for FG when last trial is last session trial.
            next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
            wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)] # wait for pokes
            next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)] # first wait for poke
            next_centre_poke_times[i] = next_wait_for_poke['Time end'].values[0] # time of first wait for poke ending

    if last_trial: # last trial in events of interest == last trial in session
        next_centre_poke_times[-1] = events_of_int['Trial end'].values[-1] + 2
    else: # last trial in events of interest != last trial in session
        event = events_of_int.tail(1)
        trial_num = event['Trial num'].values[0]
        next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
        wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)]
        next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
        next_centre_poke_times[-1] = next_wait_for_poke['Time end'].values[0]
    return next_centre_poke_times

def get_first_poke(trial_data, events_of_int): # get first poke in each trial of events of interest
    trial_numbers = events_of_int['Trial num'].unique()
    next_centre_poke_times = np.zeros(events_of_int.shape[0])
    events_of_int = events_of_int.reset_index(drop=True)
    for trial_num in trial_numbers:
        event_indx_for_that_trial = events_of_int.loc[(events_of_int['Trial num'] == trial_num)].index
        trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num)]
        wait_for_pokes = trial_events.loc[(trial_events['State type'] == 2)]
        next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
        #next_centre_poke_times[event_indx_for_that_trial] = next_wait_for_poke['Time end'].values[0]-1 #why -1 in FG code?
        next_centre_poke_times[event_indx_for_that_trial] = next_wait_for_poke['Time end'].values[0]
    return next_centre_poke_times

def get_reward_time(trial_data, events_of_int):
    ''' returns the time of the reward for each event of interest
    in trials where the mouse got a reward. If the mouse did not get a reward,
    the time of the reward is returned as 0.
    '''

    trial_numbers = events_of_int['Trial num'].unique()
    reward_times = np.zeros(events_of_int.shape[0])
    for trial_num in trial_numbers:
        index = events_of_int.loc[(events_of_int['Trial num']==trial_num)].index
        trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num)]
        trial_rewards = trial_events.loc[(trial_events['State type'] == 5) & (trial_events['Trial outcome'] == 1)]
        event_rewards = trial_rewards[trial_rewards['Instance in state'] == trial_rewards['Max times in state']]
        if not event_rewards.empty:
            reward_times[index] = event_rewards['Time end'].values[0]
    return reward_times

def get_outcome_time(trial_data, events_of_int): # returns the time of the outcome of the current trial, indep of rewarded or punished
    # this is FG 'get_next_reward_time'
    trial_numbers = events_of_int['Trial num'].values
    outcome_times = []
    for event_trial_num in range(len(trial_numbers)):
        trial_num = trial_numbers[event_trial_num]
        other_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num)]
        choices = other_trial_events.loc[(other_trial_events['State type'] == 5)] # 5 is the state type for choices / wait for response
        max_times_in_state_choices = choices['Max times in state'].unique() # all values in max times in state available for this trial an state type
        choice = choices.loc[(choices['Instance in state'] == max_times_in_state_choices)] # last time wait for response
        outcome_times.append(choice['Time end'].values[0])
    return outcome_times


def get_photometry_around_event(all_trial_event_times, demodulated_trace, pre_window=5, post_window=5, sample_rate=10000):
    num_events = len(all_trial_event_times)
    event_photo_traces = np.zeros((num_events, sample_rate*(pre_window + post_window)))
    for event_num, event_time in enumerate(all_trial_event_times):
        plot_start = int(round(event_time*sample_rate)) - pre_window*sample_rate
        plot_end = int(round(event_time*sample_rate)) + post_window*sample_rate
        event_photo_traces[event_num, :] = demodulated_trace[plot_start:plot_end]
    return event_photo_traces


def find_and_z_score_traces(trial_data, dff, params, norm_window=8, sort=False, get_photometry_data=True):
    response_names = ['both left and right', 'left', 'right']
    outcome_names = ['incorrect', 'correct', 'both correct and incorrect']
    title = ''
    # --------------
    # Categorising FG params.state numbers:
    # 10 = omission
    # 12 = large reward
    # 13 = large reward

    if params.state == 5.5:
        print('ERROR: code (find_and_z_score_traces) not adjusted for state 5.5!!!')
    # --------------
    # 1) State type (e.g. corresp. State name = CueDelay, WaitforResponse...)
    events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]  # State type = number of state of interest, typically 3 or 5
    #print("n = " + str(len(events_of_int)) + " events of interest")
    title = title + 'State type = ' + str(params.state) + ';'
    # --------------
    # 2) Response, trials to the left or to the right side
    if params.response != 0:    # 0 = don't care, 1 = left, 2 = right, selection of ipsi an contra side depends on fiber side
        events_of_int = events_of_int.loc[events_of_int['Response'] == params.response]
        # trials where mouse went to the right or left, dep on fiber and ipsi vs contra side
        title = title + ' Response = ' + str(params.response) + ';'
    # --------------
    # 3) First and last response:
    if params.first_choice != 0:
        events_of_int = events_of_int.loc[events_of_int['First response'] == params.first_choice]
        title = title + ' 1st response = ' + str(params.first_choice) + ';'
    if params.last_response != 0:
        events_of_int = events_of_int.loc[events_of_int['Last response'] == params.last_response]
        title = title + ' last response = ' + str(params.last_response) + ';'
    # --------------
    # 4) Outcome:
    if not params.outcome == 2:  # 2 would be if you don't care about the reward or not, hence selecting trials with an overall / final correct or incorrect outcome
        events_of_int = events_of_int.loc[events_of_int['Trial outcome'] == params.outcome]
        title = title + ' Outcome = ' + str(params.outcome) + ';'
    # --------------
    # 5) Cues / Sounds:
    if params.cue == 'high':
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == 7]
        title = title + ' Cue = high;'
    elif params.cue == 'low':
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == 1]
        title = title + ' Cue = low;'
    # --------------
    # 6) Instance in State & Repeats:
    if params.instance == -1:   # Last time in State
        events_of_int = events_of_int.loc[
            (events_of_int['Instance in state'] / events_of_int['Max times in state'] == 1)]
        title = title + ' instance (' + str(params.instance) + ') last time in state (no matter the repetitions);'
    elif params.instance == 1:  # First time in State
        events_of_int = events_of_int.loc[(events_of_int['Instance in state'] == 1)]
        title = title + ' instance (' + str(params.instance) + ') first time in state;'
#   elif params.instance == 0:                 # corresponds to I don't care
#       events_of_int = events_of_int

    if params.no_repeats == 1:                                                          # for FG code no repeats is ONLY considered when instance in state set to 1st only
        events_of_int = events_of_int.loc[events_of_int['Max times in state'] == 1]     # here, YJ: no repeats is considered indep. of which instance in state is selected
        title = title + ' no repetitions allowed (' + str(params.no_repeats) + ')'
    # --------------
    # 7) First choice directly in/correct?
    if params.first_choice_correct == 1:    # only first choice correct
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 1)]
        title = title + ' 1st choice correct (' + str(params.first_choice_correct) + ') only'
    elif params.first_choice_correct == -1: # only first choice incorrect               # what's the difference between  == 0 and .isnull? what does the diff. correspond to in bpod data?
        events_of_int = events_of_int.loc[np.logical_or(
            (events_of_int['First choice correct'] == 0), (events_of_int['First choice correct'].isnull()))]
        title = title + ' 1st choice incorrect (' + str(params.first_choice_correct) + ') only'
        if events_of_int['State type'].isin([5.5]).any():   # first incorrect choice?
            events_of_int = events_of_int.loc[events_of_int['First choice correct'].isnull()]
            # State type = 5.5 is the state where the mouse made the first incorrect choice
    # --------------
    # --------------
    events_of_int_reset = events_of_int.reset_index(drop=True)
    event_times = events_of_int[params.align_to].values # start or end of state of interest time points
    trial_nums = events_of_int['Trial num'].values
    trial_starts = events_of_int['Trial start'].values
    trial_ends = events_of_int['Trial end'].values

    other_event = np.asarray(np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))
       # for ex. time end - time start of state of interest

    state_name = events_of_int['State name'].values[0]

    last_trial = np.max(trial_data['Trial num'])                # absolutely last trial in session
    last_trial_num = events_of_int['Trial num'].unique()[-1]    # last trial that is considered in analysis meeting params requirements
    events_reset_index = events_of_int.reset_index(drop=True)   # same as above?
    last_trial_event_index = events_reset_index.loc[(events_reset_index['Trial num'] == last_trial_num)].index
        # index of the last event in the last trial that is considered in analysis meeting params requirements
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int, last_trial_num == last_trial)
    trial_starts = get_first_poke(trial_data, events_of_int)
    absolute_outcome_times = get_outcome_time(trial_data, events_of_int)
    relative_outcome_times = absolute_outcome_times - event_times

    print('     Settings: ' + title)
    if get_photometry_data == True:
        #next_centre_poke[last_trial_event_index] = events_reset_index[params.align_to].values[
         #                                             last_trial_event_index] + 1  # so that you can find reward peak
        # YJ: why is this necessary? what does it do?

        next_centre_poke_norm = next_centre_poke - event_times

        event_photo_traces = get_photometry_around_event(event_times, dff, pre_window=norm_window,
                                                         post_window=norm_window)
        norm_traces = stats.zscore(event_photo_traces.T, axis=0)

        #if len(other_event) != norm_traces.shape[1]:
         #   other_event = other_event[:norm_traces.shape[1]]
          #  print('Mismatch between #events and #other_event')
        if sort:
            arr1inds = other_event.argsort()
            sorted_other_event = other_event[arr1inds[::-1]] #sorting backwards [::-1]
            sorted_traces = norm_traces.T[arr1inds[::-1]]
            sorted_next_poke = next_centre_poke_norm[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_traces = norm_traces.T
            sorted_next_poke = next_centre_poke_norm

        time_points = np.linspace(-norm_window, norm_window, norm_traces.shape[0], endpoint=True, retstep=False,
                                  dtype=None,
                                  axis=0)
        mean_trace = np.mean(sorted_traces, axis=0)

        return time_points, mean_trace, sorted_traces, sorted_other_event, state_name, title, sorted_next_poke, trial_nums, event_times, relative_outcome_times
        #sorted other event later called reaction time

