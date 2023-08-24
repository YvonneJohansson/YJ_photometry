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
    mice = ['T6','T8'] #['TS17']

    dates = ['20230824'] # '20230814'] #,'2 # '20230511', '20230512', '20230513', '20230510', '20230511','20230512'
    all_experiments = get_all_experimental_records()

    for mouse in mice:
        for date in dates:
            experiments_to_process = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
            data = analyse_this_experiment(experiments_to_process, main_directory)


