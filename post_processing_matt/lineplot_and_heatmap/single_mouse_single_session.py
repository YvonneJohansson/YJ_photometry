import sys
sys.path.insert(1, 'C:\\Users\\Yvonne\\Documents\\YJ_photometry\\')
from post_processing_matt.utils.lineplot_and_heatmap_utils import *
#C:\Users\Yvonne\Documents\YJ_photometry\post_processing_matt\utils\lineplot_and_heatmap_utils.py
#C:\Users\Yvonne\Documents\YJ_photometry\post_processing_matt\lineplot_and_heatmap
#import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import colors, cm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import decimate
save_figs = True
reference_csv = pd.read_csv('C:\\Users\\Yvonne\\Documents\\experimental_record.csv')
generalDir = 'Z:\\users\\Yvonne\\photometry_2AC\\processed_data\\'
GenOutputDir = 'Z:\\users\\Yvonne\\photometry_2AC\\figures\\'

'''session_bpod_data.groupby(['State type', 'State name']).size()
Out[5]: 
State type  State name            
1.0         TrialStart                 279
2.0         WaitForPoke               1339
3.0         CueDelay                  1339
4.0         WaitForPortOut             279
5.0         WaitForResponse            279
5.5         First incorrect choice      99      ONLY In non-punished sessions
6.0         LeftReward                 115
7.0         RightReward                164
8.0         Drinking                   279
8.5         Leaving reward port         96       
'''

# list mice you want to create plots for
#animalIDs = ['SNL_photo71']# 'SNL_photo68', 'SNL_photo69', 'SNL_photo70', 'SNL_photo71', 'SNL_photo72', 'SNL_photo73', 'SNL_photo74', 'SNL_photo77'
animalIDs = ['TS15'] # ['YJ_03','YJ_04','YJ_05','YJ_06']
dates = ['20230512'] #, '20230427', '20230430', '20230501', '20230502', '20230503'] # ['20230216', '20230220', '20230222', '20230224']
alignment = 'movement'
psycho = 0

# reward: state of interest = 5, time end; wait for response end = reward (= start of drinking)
# cue: state_type_of_interest = 3; time start; no_repeats = 0 (cue delay is often repeated many times, take only the last one)
# movement: state_type_of_interest = 5, time start;

params = {'state_type_of_interest': 5, # 5.5 = first incorrect choice
    'outcome': 1, # correct or incorrect: 0 = incorrect, 1 = correct, 2 = both  *based on first choice correct, MW. Note: different for FG code*
    'last_outcome': 0,  # NOT USED CURRENTLY
    'no_repeats' :1, # 0 = dont care, 1 = state only entered once,
    'last_response':0, # trial before: 0 = dont care. 1 = left, 2 = right  # this doesn't seem to work, same plot no matter which number
    'align_to' : 'Time end', # time end or time start
    'instance': 0, # only for no repeats = 0, -1 = last instance, 1 = first instance
    'plot_range': [-6, 6],
    'first_choice_correct': 1, # useful for non-punished trials 0 = dont care, 1 = only correct trials, (-1 = incorrect trials)
    'cue': None}

if alignment == 'movement':
    params['align_to'] = 'Time start'
    params['state_type_of_interest'] = 5
    params['no_repeats'] = 1
    print('Traces aligned to ' + alignment)
    print(params)
elif alignment == 'reward':
    params['align_to'] = 'Time end'
    params['state_type_of_interest'] = 5
    params['no_repeats'] = 1
    print('Traces aligned to ' + alignment)
elif alignment == 'cue':
    params['align_to'] = 'Time start'
    params['state_type_of_interest'] = 3
    params['no_repeats'] = 0
    params['instance'] = -1
    print('Traces aligned to ' + alignment)

if psycho == 1:
    alignment = alignment + '_psychometric'

'''Plot parameters'''
error_bars= 'sem'
xlims = [-2, 3]
ylims = [-1.1, 1.1]
cue_vline = 0
all_animals = []

for animalID in animalIDs:
    inputDir = generalDir + animalID
    dates_dict = {}
    for date in dates:
        session_bpod_data = pd.read_pickle(inputDir + '\\' + animalID + '_' + date + '_restructured_data.pkl')
        photometry_trace = np.load(inputDir + '\\' + animalID + '_' + date + '_smoothed_signal.npy')
        fiber_side = reference_csv[(reference_csv['mouse_id'] == animalID) &
                               (reference_csv['date'] == date)]['fiber_side'].values[0] #== int(date))]['fiber_side'].values[0]
        if psycho == 0:
            dates_dict[date] = photometry_data(fiber_side, session_bpod_data, params, photometry_trace)
        elif psycho == 1:
            dates_dict[date] = photometry_data_psychometric(fiber_side, session_bpod_data, params, photometry_trace)

    # ------------------
        recording_site = reference_csv[(reference_csv['mouse_id'] == animalID) &
                                       (reference_csv['date'] == date)]['recording_site'].values[0]

        #make_plot_and_heatmap(animals_dict[animalID][date], error_bar_method=error_bars,
         #                 mean_across_mice=False, xlims=xlims, cue_vline=cue_vline, psycho=psycho)
        #plt.suptitle(animalID + ' (' + date + ') aligned to ' + alignment + ' (' + recording_site + '_' + fiber_side + ')')



    #------------------
    all_animals.append(dates_dict)

animals_dict = {animalID: dict for animalID, dict in zip(animalIDs, all_animals)}
#print('animals_dict:')
#print(animals_dict)

for animalID in animalIDs:
    for date in dates:
        #print(animals_dict[animalID][date])
        recording_site = reference_csv[(reference_csv['mouse_id'] == animalID) &
              (reference_csv['date'] == date)]['recording_site'].values[0]
        fiber_side = reference_csv[(reference_csv['mouse_id'] == animalID) &
                                   (reference_csv['date'] == date)]['fiber_side'].values[0]
        make_plot_and_heatmap(animals_dict[animalID][date], error_bar_method=error_bars,
                              mean_across_mice=False, xlims=xlims, cue_vline=cue_vline,psycho=psycho)
        plt.suptitle(animalID + ' (' + date + ') aligned to ' + alignment + ' (' + recording_site + '_' + fiber_side + ')' )

        # Save fig
        if save_figs == True:

            outputDir = GenOutputDir + animalID
            if not os.path.isdir(outputDir):
                os.mkdir(outputDir)

            if params['state_type_of_interest'] == 3:
                plt.savefig(
                    outputDir + '/' + animalID + '_' + date + '_' + recording_site + '_' + alignment + '.png')
            elif params['state_type_of_interest'] == 5:
                if params['align_to'] == 'Time end':
                    plt.savefig(
                        outputDir + '/' + animalID + '_' + date + '_' + recording_site + '_' + alignment + '.png')
                elif params['align_to'] == 'Time start':
                    plt.savefig(
                        outputDir + '/' + animalID + '_' + date + '_' + recording_site + '_' + alignment + '.png')
            elif params['state_type_of_interest'] == 2:
                    plt.savefig(
                        outputDir + '/' + animalID + '_' + date + '_' + recording_site + '_' + 'aligned_trial_start' + '.png')

            elif params['state_type_of_interest'] == 8:
                    plt.savefig(
                        outputDir + '/' + animalID + '_' + date + '_' + recording_site + '_' + alignment + '.png')


