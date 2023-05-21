import sys

import matplotlib.pyplot as plt

sys.path.insert(1, 'C:\\Users\\Yvonne\\Documents\\YJ_photometry\\yvonne_analysis')

import pandas as pd
from yvonne_basic_functions import *

# main_directory = r'Z:\users\Yvonne\photometry_2AC'
main_directory = 'Z:\\users\\Yvonne\\photometry_2AC\\'

# function by YJ to analyse a single session of a single mouse

if __name__ == '__main__':
    mouse_ids = ['TS3']
    dates = ['20230303']
    all_experiments = get_all_experimental_records()

    for mouse_id in mouse_ids:
        for date in dates:
            experiments_to_process = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
            # print('Experiments to process: ' + experiments_to_process)
            analyse_this_experiment(experiments_to_process, main_directory)

            #data = get_SessionData(main_directory, mouse_id, date, 'right', 'tail')
            #plt.plot(data.choice.contra_data.time_points,data.choice.contra_data.mean_trace)
            #plt.xlim([-2, 3])
            #plt.text(-2, -0.6, data.choice.contra_data.title, ha='center', va='center')
            #plt.show()