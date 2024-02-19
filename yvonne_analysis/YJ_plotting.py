import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
from tqdm import tqdm
import matplotlib as mpl
#import cairocffi as cairo
from scipy.signal import decimate

from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import numpy as np
from yvonne_basic_functions import *



def make_y_lims_same(ylim_A, ylim_B):
    ylim_min = min(ylim_A[0], ylim_B[0])
    ylim_max = max(ylim_A[1], ylim_B[1])
    return ylim_min, ylim_max

def calculate_error_bars(mean_trace, data, error_bar_method='sem'):
    if error_bar_method == 'sem':
        sem = stats.sem(data, axis=0)
        lower_bound = mean_trace - sem
        upper_bound = mean_trace + sem
    elif error_bar_method == 'ci':
        lower_bound, upper_bound = bootstrap(data, n_boot=1000, ci=68)
    return lower_bound, upper_bound


def plot_avg_sem(aligned_traces, fig, ax, ax_overlay, y_range, error_bar_method='sem', y_label='', top_label='', color='default'):

    # Trace average and sem:
    mean_trace = decimate(aligned_traces.mean_trace, 10)

    # include only traces where mean larger than 0.5 during first 2 seconds


    time_points = decimate(aligned_traces.time_points, 10)
    traces = decimate(aligned_traces.sorted_traces, 10)
    if color == 'default':
        ax.plot(time_points, mean_trace, lw=1.5, color='#3F888F')
        ax_overlay.plot(time_points, mean_trace, lw=1.5, color='#3F888F')
    else:
        ax.plot(time_points, mean_trace, lw=1.5, color=color)
        ax_overlay.plot(time_points, mean_trace, lw=1.5, color=color)

    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace, traces,
                                                                error_bar_method=error_bar_method)
        ax.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                         facecolor='#7FB5B5', linewidth=0)

    ax.axvline(0, color='#808080', linewidth=0.5)
    ax.set_xlim([-2,3])
    ax.set_ylim(y_range)
    ax.yaxis.set_ticks(np.arange(y_range[0], y_range[1] + 1, 1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(y_label + 'z-score')
    ax.set_title(top_label, fontsize=14, rotation="horizontal", va="center", color='#3F888F')
                  #color='#3F888F') #y=np.mean(y_range) va="center"
    right_axis = ax.spines["right"]
    right_axis.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)

    ax_overlay.axvline(0, color='#808080', linewidth=0.5)
    ax_overlay.set_xlim([-0.5,1.5])
    ax_overlay.set_ylim(y_range)
    ax_overlay.yaxis.set_ticks(np.arange(y_range[0], y_range[1] + 1, 1))
    ax_overlay.set_xlabel('Time (s)')
    ax_overlay.set_ylabel(y_label + 'z-score')
    #ax_overlay.set_title(top_label, fontsize=14, rotation="horizontal", va="center", color='#3F888F')
    right_axis = ax_overlay.spines["right"]
    right_axis.set_visible(False)
    top = ax_overlay.spines["top"]
    top.set_visible(False)


def add_plot_one_side(one_side_data, fig, ax1, ax2, y_range, hm_range, error_bar_method='sem', sel_color = 'default'):
    # Trace average and sem:
    if one_side_data.mean_trace.shape[0] != 0:
    #if one_side_data.mean_trace.shape != np.empty((0, 0)): # sometimes there is no photometry data for a specific situation (e.g. reward incorrect ipsi etc)
        mean_trace = decimate(one_side_data.mean_trace, 10)
        time_points = decimate(one_side_data.time_points, 10)
        traces = decimate(one_side_data.sorted_traces, 10)

        if traces.shape[0] > 5: # plotting only works if there is more than one trace, >5 because it looks like crap otherwise
            if sel_color == 'default':
                sel_color = '#3F888F'

            ax1.plot(time_points, mean_trace, lw=1.5, color=sel_color)  # default: '#3F888F'

            if error_bar_method is not None:
                error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace, traces,
                                                                    error_bar_method=error_bar_method)
                ax1.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.3,
                             facecolor=sel_color, linewidth=0) # default: '#7FB5B5'
    else:
        print('no data for add_plot_one_side')
    return ax1, ax2


def plot_one_side(one_side_data, fig, ax1, ax2, y_range, hm_range, error_bar_method='sem', sort=False,
                  white_dot='default', y_label='', top_label=''):
    # Trace average and sem:
    mean_trace = decimate(one_side_data.mean_trace, 10)
    time_points = decimate(one_side_data.time_points, 10)
    traces = decimate(one_side_data.sorted_traces, 10)

    if np.max(mean_trace) > y_range[1]:
        y_range = (y_range[0], np.ceil(np.max(mean_trace)))

    if traces.shape[0] > 1: # plotting only works if there is more than one trace

        if y_label == 'SOR Contra ':
            color_mean = '#e377c2'
            color_sem = '#e377c2'
            alpha = 0.5
        else:
            color_mean = '#3F888F'
            color_sem = '#7FB5B5'
            alpha = 1

        ax1.plot(time_points, mean_trace, lw=1.5, color=color_mean)

        if error_bar_method is not None:
            error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace, traces,
                                                                error_bar_method=error_bar_method)
            ax1.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=alpha,
                         facecolor=color_sem, linewidth=0)

        y_total = y_range[1] - y_range[0]
        #print('y_total:' + str(y_total) + ', set: ' + str(np.mean(y_range) - 1 / 4 * y_total))
        ax1.axvline(0, color='#808080', linewidth=0.5, ls = 'dashed')
        ax1.set_xlim(one_side_data.params.plot_range)
        #ax1.set_ylim(-1,1.5)
        ax1.set_ylim(y_range)
        ax1.yaxis.set_ticks(np.arange(y_range[0], y_range[1] + 1, 1))
        ax1.set_xlabel('Time (s)')
        yaxis = ax1.set_ylabel(y_label + 'z-score')

        #title = ax1.set_title(top_label, fontsize=14, color='#3F888F')
        #offset = np.array([0, 0.0])
        #title.set_position(yaxis.get_position() + offset)
        #title.set_rotation("vertical")

        ax1.set_title(top_label, x=-0.5, y=0.2*np.mean(y_range), fontsize=14, rotation="vertical",
                     color='#3F888F') #y=np.mean(y_range) va="center"
        right_axis = ax1.spines["right"]
        right_axis.set_visible(False)
        top = ax1.spines["top"]
        top.set_visible(False)

        # ----------------------------------------------------------
        # Heatmap:

        if white_dot == 'reward':
            white_dot_point = one_side_data.outcome_times
        else:
            white_dot_point = one_side_data.reaction_times



        if sort:
            arr1inds = white_dot_point.argsort()
            one_side_data.reaction_times = one_side_data.reaction_times[arr1inds[::-1]]
            one_side_data.outcome_times = one_side_data.outcome_times[arr1inds[::-1]]
            one_side_data.sorted_traces = one_side_data.sorted_traces[arr1inds[::-1]]
            one_side_data.sorted_next_poke = one_side_data.sorted_next_poke[arr1inds[::-1]]

        # nr. trials to show: one_side_data.sorted_traces.shape[0]

        heat_im = ax2.imshow(one_side_data.sorted_traces, aspect='auto',
                             extent=[-10, 10, one_side_data.sorted_traces.shape[0], 0], cmap='viridis')

        ax2.axvline(0, color='w', linewidth=1)

        if white_dot == 'reward':
            ax2.scatter(one_side_data.outcome_times,
                        np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=0.15)
        elif white_dot == None:
            pass
        else:
            ax2.scatter(one_side_data.reaction_times,
                        np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=0.15)

        # next trial
        if white_dot is not None: #RTC has no sorted next poke
            ax2.scatter(one_side_data.sorted_next_poke,
                    np.arange(one_side_data.sorted_next_poke.shape[0]) + 0.5, color='b', s=0.5)

        ax2.tick_params(labelsize=10)
        ax2.set_xlim(one_side_data.params.plot_range)
        ax2.set_ylim([one_side_data.sorted_traces.shape[0], 0])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Trial (sorted)')

        if y_range:
            norm = colors.Normalize(vmin=hm_range[0], vmax=hm_range[1])
            heat_im.set_norm(norm)

        cb = fig.colorbar(heat_im, ax=ax2, orientation="vertical", fraction=0.1)
        cb.ax.set_title('z-score', fontsize=9, pad=4)

        return heat_im


def heat_map_and_mean_SingleSession(SessionData, error_bar_method='sem', sort=False, x_range=[-2, 3], white_dot='default'):

    if SessionData.protocol == 'SOR' and SessionData.fiber_side == 'right':
        rows = 3 #6
        columns = 6
        height = 8.25 # * 2
        width = 10 * 1.5
    else:
        rows = 3
        columns = 4
        width = 10
        height = 8.25

    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(width, height))  # width, height
    #fig, axs = plt.subplots(nrows=rows, ncols=4, figsize=(11, height)) # width, height
    fig.tight_layout(pad=4)
    #fig.tight_layout(pad=2.1, rect=[0,0,0.05,0])  # 2.1)
    font = {'size': 10}

    # vorher:
    #plt.rcParams["pdf.fonttype"] = 42
    #plt.rcParams["ps.fonttype"] = 42
    #plt.rcParams["font.family"] = "Arial"
    #plt.rcParams["font.size"] = 10

    # neu:

    plt.rc('font', family='Arial', size=12)
    plt.rcParams.update({'font.size': 12})
    #mpl.use('cairo')
        # plt.rcParams['figure.figsize'] = [10, 5]
         #   import matplotlib.font_manager as fm
        # font = fm.FontProperties(family = 'Arial', size = 12)
        # später: plt.title('xxx', fontproperties = font)


    x_range = x_range

    alignements = ['cue', 'movement', 'reward']

    # ----------------------------------------------------------------
    # getting min and max value across all data to be plotted:
    y_minmax = []
    hm_minmax = []
    for alignement in alignements:
        if alignement == 'cue':
            aligned_data = SessionData.cue
        elif alignement == 'movement':
            aligned_data = SessionData.choice
        elif alignement == 'reward':
            aligned_data = SessionData.reward

        aligned_data.ipsi_data.params.plot_range = x_range
        aligned_data.contra_data.params.plot_range = x_range

        y_minmax.append(np.min(aligned_data.ipsi_data.mean_trace))
        y_minmax.append(np.max(aligned_data.ipsi_data.mean_trace))
        y_minmax.append(np.min(aligned_data.contra_data.mean_trace))
        y_minmax.append(np.max(aligned_data.contra_data.mean_trace))

        if data.protocol == 'LRO' or data.protocol == 'LargeRewards':
            y_minmax.append(np.max(SessionData.reward.contra_data_LR.mean_trace))
            y_minmax.append(np.min(SessionData.reward.contra_data_LR.mean_trace))
        if data.protocol == 'LRO' or data.protocol == 'Omissions':
            y_minmax.append(np.min(SessionData.reward.contra_data_O.mean_trace))
            y_minmax.append(np.min(SessionData.reward.ipsi_data_O.mean_trace))

        if SessionData.reward.contra_data_incorrect.mean_trace.shape[0] is not 0:
            y_minmax.append(np.max(SessionData.reward.contra_data_incorrect.mean_trace)) # activate again
            y_minmax.append(np.min(SessionData.reward.contra_data_incorrect.mean_trace))  # activate again
        #y_minmax.append(np.max(SessionData.reward.ipsi_data_incorrect.mean_trace))
        #y_minmax.append(np.min(SessionData.reward.ipsi_data_incorrect.mean_trace))


        hm_minmax.append(np.min(aligned_data.ipsi_data.sorted_traces))
        hm_minmax.append(np.max(aligned_data.ipsi_data.sorted_traces))
        hm_minmax.append(np.min(aligned_data.contra_data.sorted_traces))
        hm_minmax.append(np.max(aligned_data.contra_data.sorted_traces))

    ylim_max = math.ceil(max(y_minmax))
    ylim_min = math.floor(min(y_minmax))
    y_range = (ylim_min, ylim_max)
    hm_max = max(hm_minmax)
    hm_min = min(hm_minmax)
    hm_range = (hm_min, hm_max)
    # print('total heatmap range: ' + str(hm_range))

    # ----------------------------------------------------------------
    # Plotting
    for alignement in alignements:
        if alignement == 'cue':
            aligned_data = SessionData.cue
            row = 0
        elif alignement == 'movement':
            aligned_data = SessionData.choice
            row = 1
        elif alignement == 'reward':
            aligned_data = SessionData.reward
            row = 2

        c_column = 0 # contra column for 2AC
        i_column = 2 # ipsi column for 2AC
        if SessionData.protocol == 'SOR' and SessionData.fiber_side == 'right':
            c_column = 2 # contra column for 2AC if SOR plots on the left
            i_column = 4 # ipsi column for 2AC if SOR plots on the left

        aligned_data.ipsi_data.params.plot_range = x_range
        aligned_data.contra_data.params.plot_range = x_range

        if alignement == 'reward':
            add_incorrect = True
            legend_text = []
            if data.protocol == 'LRO' or data.protocol == 'LargeRewards':
                contra_LRO_traces, legend = add_plot_one_side(SessionData.reward.contra_data_LR, fig, axs[row, c_column], axs[row, c_column + 1], y_range, hm_range, error_bar_method=error_bar_method, sel_color = '#e377c2')
                ipsi_LRO_traces, legend = add_plot_one_side(SessionData.reward.ipsi_data_LR, fig, axs[row, i_column], axs[row, i_column +1], y_range, hm_range, error_bar_method=error_bar_method, sel_color = '#e377c2')
                legend_text.append('LR')
                add_incorrect = False
            if data.protocol == 'LRO' or data.protocol == 'Omissions':
                contra_LRO_traces, legend = add_plot_one_side(SessionData.reward.contra_data_O, fig, axs[row, c_column], axs[row, c_column + 1], y_range, hm_range, error_bar_method=error_bar_method, sel_color = '#9467bd')
                ipsi_LRO_traces, legend = add_plot_one_side(SessionData.reward.ipsi_data_O, fig, axs[row, i_column], axs[row, i_column +1], y_range, hm_range, error_bar_method=error_bar_method, sel_color = '#9467bd')
                add_incorrect = False
                legend_text.append('O')

            add_incorrect_legend = False
            if add_incorrect:
                if SessionData.reward.contra_data_incorrect.mean_trace.shape[0] != 0:
                    contra_incorrect, legend = add_plot_one_side(SessionData.reward.contra_data_incorrect, fig, axs[row, c_column], axs[row, c_column + 1], y_range, hm_range, error_bar_method=error_bar_method, sel_color = '#DC143C')
                    add_incorrect_legend = True
                else:
                    print('no SessionData.reward.contra_data_incorrect.mean_trace trials to plot' + str(SessionData.reward.contra_data_incorrect.mean_trace.shape))

                if SessionData.reward.ipsi_data_incorrect.mean_trace.shape[0] != 0:
                    ipsi_incorrect, legend = add_plot_one_side(SessionData.reward.ipsi_data_incorrect, fig, axs[row, i_column], axs[row, i_column + 1], y_range, hm_range, error_bar_method=error_bar_method, sel_color = '#DC143C')
                    add_incorrect_legend = True
                else:
                    print('no SessionData.reward.ipsi_data_incorrect.mean_trace trials to plot' + str(SessionData.reward.ipsi_data_incorrect.mean_trace.shape))

                if add_incorrect_legend:
                    legend_text.append('incorrect')
            axs[row, 0].legend(legend_text, loc='upper right', fontsize=8, frameon=False)


        contra_heatmap = plot_one_side(aligned_data.contra_data, fig, axs[row, c_column], axs[row, c_column +1], y_range, hm_range,   # axs[row, 0], axs[row, 1]
                                       error_bar_method=error_bar_method, sort=sort, white_dot=white_dot,
                                       y_label='Contra ', top_label=alignement)
        ipsi_heatmap = plot_one_side(aligned_data.ipsi_data, fig, axs[row, i_column], axs[row, i_column + 1], y_range, hm_range,   #axs[row, 2], axs[row, 3]
                                     error_bar_method=error_bar_method, sort=sort, white_dot=white_dot,
                                     y_label='Ipsi ', top_label='')




        text = SessionData.mouse + '_' + SessionData.date + '_' + SessionData.recording_site + '_' + \
               SessionData.fiber_side + ', protocol: ' + SessionData.protocol + SessionData.protocol_info + ', performance: ' + str("%.2f" % SessionData.performance) + '% in n = ' + str("%.0f" % SessionData.nr_trials) + ' trials'

        axs[0, 0].text(x_range[0]-2.8, y_range[1]+0.5, text, fontsize=12)

    if SessionData.protocol == 'SOR' and SessionData.fiber_side == 'right':
        for alignement in alignements:
            if alignement == 'cue':
                aligned_data = SessionData.SOR_cue
                row = 0 #3
            elif alignement == 'movement':
                aligned_data = SessionData.SOR_choice
                row = 1 # 4
            elif alignement == 'reward':
                aligned_data = SessionData.SOR_reward
                row = 2 # 5

            aligned_data.contra_data.params.plot_range = x_range

            if alignement == 'reward':
                legend_text = []
                color_mean = '#e377c2'
                color_sem = '#e377c2'
                alpha = 0.3
                contra_incorrect, legend = add_plot_one_side(SessionData.reward.contra_data_incorrect, fig, axs[row, 0],
                                                             axs[row, 1], y_range, hm_range,
                                                             error_bar_method=error_bar_method, sel_color='#DC143C')

                legend_text.append('incorrect')
                axs[row, 0].legend(legend_text, loc='upper right', fontsize=8, frameon=False)


            contra_heatmap = plot_one_side(aligned_data.contra_data, fig, axs[row, 0], axs[row, 1], y_range, hm_range,
                                           error_bar_method=error_bar_method, sort=sort, white_dot=white_dot,
                                           y_label='SOR Contra ', top_label=alignement)


    return fig


def heat_map_and_mean_SingleSession_ExtraSession(Session_data, Extra_trial_data, Extra_df, extra_session, error_bar_method='sem', sort=True, x_range=[-2, 3], white_dot='default'):

    rows = 2
    height = 5.5
    if extra_session == 'Airpuff':
        rows = 3
        height = 8.25


    fig, axs = plt.subplots(nrows=rows, ncols=2, figsize=(5.5, height))  # width, height
    fig.tight_layout(pad=4)
    font = {'size': 10}
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10

    alignement = 'movement'
    data = Session_data
    if data.protocol == 'SOR':
        aligned_data = data.SOR_choice.contra_data
    else:
        aligned_data = data.choice.contra_data

    aligned_data.params.plot_range = x_range

    ylim_min = math.floor(np.min(aligned_data.mean_trace))
    ylim_max = math.ceil(np.max(aligned_data.mean_trace))
    hm_max = np.max(aligned_data.sorted_traces)
    hm_min = np.min(aligned_data.sorted_traces)
    y_range = (ylim_min, ylim_max)
    hm_range = (hm_min, hm_max)
    aligned_data.params.plot_range = x_range

    # movement response in preceding session:
    contra_heatmap = plot_one_side(aligned_data, fig, axs[0, 0], axs[0, 1], y_range, hm_range,
                                   error_bar_method=error_bar_method, sort=sort, white_dot=white_dot,
                                   y_label='Contra ', top_label=alignement)

    # Random_Tone_Clouds:
    #print(trial_data['Sound type'].unique())
    #if RTC_trial_data['Sound type'].unique() == 2:
     #   curr_label = 'RWN'
    #else:
     #   curr_label = 'RTC'
    curr_label = extra_session

    if curr_label == 'Airpuff': # splitting shorter and longer airpuffs
        airpuff_data = ZScoredTraces_Airpuff(Extra_trial_data, Extra_df, x_range)

        Airpuff_heatmap = plot_one_side(airpuff_data.shortAirpuff, fig, axs[1, 0], axs[1, 1], y_range, hm_range,
                                      error_bar_method=error_bar_method, sort=False, white_dot=None,
                                      y_label='', top_label='short ' + curr_label)

        Airpuff_heatmap = plot_one_side(airpuff_data.longAirpuff, fig, axs[2, 0], axs[2, 1], y_range, hm_range,
                                      error_bar_method=error_bar_method, sort=False, white_dot=None,
                                      y_label='', top_label='long ' + curr_label)

    else: # RTC, RWN:
        Extra_data = ZScoredTraces_RTC(Extra_trial_data, Extra_df, x_range)

        Extra_heatmap = plot_one_side(Extra_data, fig, axs[1, 0], axs[1, 1], y_range, hm_range,
                                   error_bar_method=error_bar_method, sort=False, white_dot=None,
                                   y_label='', top_label=curr_label)


    text = data.mouse + '_' + data.date + '_' + data.recording_site + '_' + \
               data.fiber_side + ', protocol: ' + data.protocol + data.protocol_info + ', performance: ' + str("%.2f" % data.performance) + '% in n = ' + str("%.0f" % data.nr_trials) + ' trials vs ' + curr_label

    axs[0, 0].text(x_range[0]-2.8, y_range[1]+0.5, text, fontsize=8)

    return fig


def CueResponses_DMS_vs_TS(all_experiments, mice, locations, main_directory, error_bar_method='sem', x_range=[-2, 3], minPerformance=75):

    fig, axs = plt.subplots(nrows=len(mice)+1, ncols=4, figsize=(11, 8.25))
    fig.tight_layout(pad=4)
    # fig.tight_layout(pad=2.1, rect=[0,0,0.05,0])  # 2.1)
    font = {'size': 10}
    plt.rc('font', **font)
    x_range = x_range
    y_range = [-1, 3]
    alignements = ['cue', 'movement', 'reward']
    included_exps = pd.DataFrame()
    print('Mice to look at: ' + str(mice))

    for mouse in mice:
        row = mice.index(mouse)
        for location in locations:
            experiments = all_experiments[(all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == location)]
            print('Potential experiments for ' + mouse + ': ' + str(len(experiments)) + ' in ' + location)
            if location == 'DMS':
                color = '#e377c2'
            elif location == 'TS':
                color = 'default'
            # Loop backwards through experiments to find the most recent one that meets the criteria
            for index, experiment in experiments.iterrows():
                date = experiment['date']
                fiber_side = experiment['fiber_side']
                try:
                    data = get_SessionData(main_directory, mouse, date, fiber_side, location)
                    print('Next:' + mouse +' '+ date + ' ' + location)
                    if np.max(data.cue.contra_data.mean_trace) > 0.5 or np.max(data.choice.contra_data.mean_trace) > 0.5:
                        try:
                            if data.performance > minPerformance and data.protocol == '2AC':
                                print('> Plotting: ' + mouse + '_' + location + '_' + date)
                                column = locations.index(location)
                                aligned_traces = data.choice.contra_data
                                print('now: row ' + str(row) + ' column ' + str(column))
                                ax=axs[row, column]
                                ax_overlay=axs[row, 2]
                                plot_avg_sem(aligned_traces, fig, ax, ax_overlay, y_range, error_bar_method='sem', y_label='contra choice', top_label= mouse + '_' + location,color=color)

                            else:
                                print('   >> Not plotting: as ' + str("%.0f" % data.performance) + '% performance in protocol ' + data.protocol + ' from ' + date)
                        except AttributeError:
                            print('   >> No performance or protocol data found for ' + mouse + '_' + date + '_' + fiber_side + '_' + location)
                    else:
                        print('   >> Not plotting: as too small cue or movement response in ' + date)
                except OSError:
                    print(
                    '       >> No trial data found for ' + mouse + '_' + date + '_' + fiber_side + '_' + location)

    return fig







# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# function by YJ to analyse a single session of a single mouse

if __name__ == '__main__':
    mice = ['TS26'] #,'T5','T6','T8'] #,'TS20']['TS20','TS21'] #
    dates = ['20231004'] # ['20230922'] #['20230904'] #,'20230513']['20230513','20230514'] #'20230728','20230731','20230802','20230808','20230809'
    recording_site = 'TS'
    fiber_side = 'right'
    exclude_protocols = ['psychometric','LRO']

    # --------------------------------------
    plot = 1 # 1: SingleSession, 2: Random_Tone_Clouds, 3: CueResponses_DMS_vs_TS, 4: TimeSeries, 5: Random_WN, 6: Airpuff
    # --------------------------------------

    all_experiments = get_all_experimental_records()
    main_directory = 'Z:\\users\\Yvonne\\photometry_2AC\\'
    if plot == 1: # 'SingleSession':
        print('SingleSession')
        for mouse in mice:
            for date in dates:
                experiment = all_experiments[
                    (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
                print('   > Now analysing:' + mouse + ' ' + date)
                fiber_side = experiment['fiber_side'].values[0]
                recording_site = experiment['recording_site'].values[0]
                data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)
                text = mouse + '_' + date + '_' + recording_site + '_' + fiber_side + '_' + data.protocol
                figure = heat_map_and_mean_SingleSession(data, error_bar_method='sem', sort=True, x_range=[-2, 3], white_dot='default')

                #figure = heat_map_and_mean_SingleSession(data, error_bar_method='sem', sort=True, x_range=[-2, 3],
                 #                                   white_dot='default')
                #canvas = FigureCanvasSVG(figure)
                #canvas.print_svg(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.svg', dpi=600)
                #plt.savefig(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.png', bbox_inches="tight", dpi=600)
                #plt.savefig(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.pdf', bbox_inches="tight", dpi=600)

                plt.savefig(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.pdf', transparent=True, dpi=300)
                #with PdfPages(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.pdf') as pdf:
                 #   pdf.savefig(figure, transparent=True, bbox_inches="tight", dpi=600)

                plt.show()

    if plot == 2:
        print('Random_Tone_Clouds')
        protocol = 'Random_Tone_Clouds'
        for mouse in mice:
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            search_trial_data = '_RTC_restructured_data.pkl'
            search_df_data = '_RTC_smoothed_signal.npy'
            files_in_path = os.listdir(trial_data_path)
            trial_data_files = [file for file in files_in_path if search_trial_data in file]
            df_data_files = [file for file in files_in_path if search_df_data in file]

            nr_files = len(trial_data_files)

            for i in range(0,nr_files):
                trial_data_name = trial_data_files[i]
                trial_data = pd.read_pickle(trial_data_path + trial_data_name)
                df_name = df_data_files[i]
                df =np.load(trial_data_path + df_name)

                date = df_data_files[i].split('_')[1]
                # get movement signal from same day session:
                experiment = all_experiments[
                    (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
                fiber_side = experiment['fiber_side'].values[0]
                recording_site = experiment['recording_site'].values[0]
                data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)

                print(mouse + '_' + date + '_' + data.protocol + '_&_Random_Tone_Clouds')
                figure = heat_map_and_mean_SingleSession_ExtraSession(data, trial_data, df, extra_session = 'RTC', error_bar_method='sem', sort=True, x_range=[-2, 3], white_dot='default')
                canvas = FigureCanvasSVG(figure)
                text = mouse + '_' + date + '_' + recording_site + '_' + fiber_side + '_RTC_vs_' + data.protocol
                with PdfPages(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.pdf') as pdf:
                    pdf.savefig(figure, transparent=True, bbox_inches="tight", dpi=600)

                plt.show()


    if plot == 3:
        print(mice)
        locations = ['TS','DMS']
        figure = CueResponses_DMS_vs_TS(all_experiments, mice, locations, main_directory, error_bar_method='sem', x_range=[-2, 3], minPerformance=60)
        plt.show()


    if plot == 4:
        print('TimeSeries')
        alignements = ['cue', 'movement', 'reward']
        p_row = -1
        nr_rows = len(mice)
        if nr_rows < 2:
            nr_rows = nr_rows + 1

        fig, axs = plt.subplots(nrows=nr_rows, ncols=3, figsize=(3 * 4, nr_rows * 4))
        fig.tight_layout(pad=8)

        for mouse in mice:
            print('-----------------')
            print('NOW: ' + mouse)

            experiments = all_experiments[(all_experiments['mouse_id'] == mouse) & (all_experiments['fiber_side'] == fiber_side) &
                                          (all_experiments['recording_site'] == recording_site) & (all_experiments['include'] != 'no')]

            n_exp = len(experiments)

            excl_date = []
            for excl in exclude_protocols:
                mask = experiments
                exclude_experiments = search(mask, excl)
                try:
                    excl_date.append(exclude_experiments['date'].values[0])
                except IndexError:
                    print('')

            for j in excl_date:
                experiments = experiments[experiments['date'] != j]
                print(j)

            n_exp = len(experiments)
            print('Number of experiments: ' + str(n_exp))

            p_row = p_row + 1
            colors = plt.cm.viridis(np.linspace(0, 1, n_exp))

            experiments.sort_values('date', inplace=True)
            #experiments.sort_values(by = ['date'], ascending = True)  # start with earliest date, no matter sequence in csv file
            experiments.reset_index(drop=True, inplace=True)
            performances = []

            for i, row in experiments.iterrows():
                date = row['date']

                folder = main_directory + 'YJ_results\\' + mouse + '\\'
                filename = folder + mouse + '_' + date + '_' + fiber_side + '_' + recording_site + '_' + 'aligned_traces.p'
                with open(filename, 'rb') as input_file:
                    SessionData = pickle.load(input_file)
                    try:
                        protocol = SessionData.protocol
                    except AttributeError:
                        print('     >> No protocol for ' + date)
                    csv_protocol = all_experiments[(all_experiments['date'] == SessionData.date) & (
                            all_experiments['mouse_id'] == SessionData.mouse)]['experiment_notes'].values[0]
                    if protocol != '2AC':
                        print('       >> Protocol = ' + protocol + ' for ' + mouse + '_' + date + ' while csv: ' + csv_protocol)

                for alignement in alignements:
                            if alignement == 'cue':
                                aligned_data = SessionData.cue.contra_data
                                performances.append(str(i+1) + '. ' + str("%.0f" % SessionData.performance) + '% ') # + SessionData.protocol + ' ' + date)
                                col = 0
                            elif alignement == 'movement':
                                aligned_data = SessionData.choice.contra_data
                                col = 1
                            elif alignement == 'reward':
                                aligned_data = SessionData.reward.contra_data
                                col = 2

                            #print('i = ' + str(i) + ' p_row = ' + str(p_row) + ' col = ' + str(col))

                            mean_trace = decimate(aligned_data.mean_trace, 10)
                            time_points = decimate(aligned_data.time_points, 10)
                            traces = decimate(aligned_data.sorted_traces, 10)

                            axs[p_row][col].plot(time_points, mean_trace, lw=1.5, color=colors[i])
                            axs[p_row][col].set_xlim(-2, 3)
                            axs[p_row][col].set_ylim(-1, 2)
                            axs[p_row][col].set_xlabel('Time (s)')
                            axs[p_row][col].set_ylabel('z-score')
                            right = axs[p_row][col].spines["right"]
                            right.set_visible(False)
                            top = axs[p_row][col].spines["top"]
                            top.set_visible(False)

                            box = axs[p_row][col].get_position()
                            axs[p_row][col].set_position([box.x0, box.y0, box.width * 0.999, box.height])
                            axs[p_row][col].text(-2, 2.4, alignement, ha='right', va='top', fontsize=9, color='#0000FF')
                            axs[p_row][2].legend(performances, bbox_to_anchor=(1.1, 0.4), frameon=False, fontsize=8)
                            axs[p_row][col].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
                            #axs[p_row][0].legend(performances, loc="upper left", frameon=False, fontsize=8)
                            axs[p_row][0].text(-2, 2.8, mouse + ': ' + recording_site + fiber_side + ', n = ' + str(n_exp),
                                           ha='center', va='center', fontsize=10)

            plt.setp(axs, yticks=[-1, 0, 1, 2], xticks=[-2, 0, 2])
        plt.show()

        canvas = FigureCanvasSVG(fig)
        with PdfPages(main_directory + 'YJ_results\\' + 'TimeSeries_' + recording_site + fiber_side + '.pdf') as pdf:
            pdf.savefig(fig, transparent=True, bbox_inches="tight", dpi=600)



    if plot == 5:
        print('Random_WN')
        protocol = 'Random_WN'
        for mouse in mice:
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            search_trial_data = '_RWN_restructured_data.pkl'
            search_df_data = '_RWN_smoothed_signal.npy'
            files_in_path = os.listdir(trial_data_path)
            trial_data_files = [file for file in files_in_path if search_trial_data in file]
            df_data_files = [file for file in files_in_path if search_df_data in file]

            nr_files = len(trial_data_files)

            for i in range(0, nr_files):
                trial_data_name = trial_data_files[i]
                trial_data = pd.read_pickle(trial_data_path + trial_data_name)
                #print(trial_data['Sound type'].unique())
                df_name = df_data_files[i]
                df = np.load(trial_data_path + df_name)

                date = df_data_files[i].split('_')[1]
                # get movement signal from same day session:
                experiment = all_experiments[
                    (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
                fiber_side = experiment['fiber_side'].values[0]
                recording_site = experiment['recording_site'].values[0]
                data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)

                print(mouse + '_' + date + '_' + data.protocol + '_&_Random_White_Noise')
                figure = heat_map_and_mean_SingleSession_ExtraSession(data, trial_data, df, extra_session='RWN', error_bar_method='sem', sort=True,
                                                             x_range=[-2, 3], white_dot='default')
                canvas = FigureCanvasSVG(figure)
                text = mouse + '_' + date + '_' + recording_site + '_' + fiber_side + '_RWN_vs_' + data.protocol
                with PdfPages(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.pdf') as pdf:
                    pdf.savefig(figure, transparent=True, bbox_inches="tight", dpi=600)

                plt.show()


    if plot == 6:
        print('Airpuff')
        protocol = 'AirpuffPhoto'
        for mouse in mice:
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            search_trial_data = '_Airpuff_restructured_data.pkl'
            search_df_data = '_Airpuff_smoothed_signal.npy'
            files_in_path = os.listdir(trial_data_path)
            trial_data_files = [file for file in files_in_path if search_trial_data in file]
            df_data_files = [file for file in files_in_path if search_df_data in file]

            nr_files = len(trial_data_files)

            for i in range(0, nr_files):
                trial_data_name = trial_data_files[i]
                trial_data = pd.read_pickle(trial_data_path + trial_data_name)
                df_name = df_data_files[i]
                df = np.load(trial_data_path + df_name)

                date = df_data_files[i].split('_')[1]
                # get movement signal from same day session:
                experiment = all_experiments[
                    (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
                fiber_side = experiment['fiber_side'].values[0]
                recording_site = experiment['recording_site'].values[0]
                data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)

                print(mouse + '_' + date + '_' + data.protocol + '_&_Airpuff')
                figure = heat_map_and_mean_SingleSession_ExtraSession(data, trial_data, df, extra_session='Airpuff', error_bar_method='sem', sort=True,
                                                             x_range=[-2, 3], white_dot='default')
                canvas = FigureCanvasSVG(figure)
                text = mouse + '_' + date + '_' + recording_site + '_' + fiber_side + '_Airpuff_vs_' + data.protocol
                with PdfPages(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.pdf') as pdf:
                    pdf.savefig(figure, transparent=True, bbox_inches="tight", dpi=600)

                plt.show()