import sys

import matplotlib.pyplot as plt

sys.path.insert(1, 'C:\\Users\\Yvonne\\Documents\\YJ_photometry\\yvonne_analysis')

import pandas as pd
from yvonne_basic_functions import *
from YJ_plotting import *

# main_directory = r'Z:\users\Yvonne\photometry_2AC'
main_directory = 'Z:\\users\\Yvonne\\photometry_2AC\\'

# function by YJ to analyse a single session of a single mouse

if __name__ == '__main__':
    mice = ['TS30', 'TS31'] #,'T5','T9'] #['TS17']
    dates = ['20231004'] # '20230814'] #,'2 # '20230511', '20230512', '20230513', '20230510', '20230511','20230512'
    set = 'SOR'

    # RTC
    #mice = ['TS3', 'TS20', 'TS21', 'TS26', 'TS33']  # 'TS29_20230927', TS34_20231102
    #dates = ['20230203', '20230512', '20230510', '20230929', '20231106']


    debug = True
    all_experiments = get_all_experimental_records()


    if set == 'SOR':
        # SOR
        mice = ['TS24', 'TS26', 'TS27', 'TS32', 'TS33', 'TS34']
        dates = ['20230929', '20230918', '20231003', '20231026', '20231102', '20231031']

        #novelty
        mice = ['TS30', 'TS31', 'TS32', 'TS33', 'TS34']
        dates = ['20231004', '20231004', '20231026', '20231026', '20231026']

        for a, mouse in enumerate(mice):
            date = dates[a]
            experiments_to_process = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
            data = analyse_this_experiment(experiments_to_process, main_directory, debug)


    else:
        for mouse in mice:
            for date in dates:
                experiments_to_process = all_experiments[
                    (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
                data = analyse_this_experiment(experiments_to_process, main_directory, debug)


