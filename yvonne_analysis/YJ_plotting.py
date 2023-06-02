import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.backends.backend_svg import FigureCanvasSVG

from tqdm import tqdm
import matplotlib
from scipy.signal import decimate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
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





def plot_one_side(one_side_data, fig, ax1, ax2, y_range, hm_range, error_bar_method='sem', sort=False,
                  white_dot='default', y_label='', top_label=''):
    # Trace average and sem:
    mean_trace = decimate(one_side_data.mean_trace, 10)
    time_points = decimate(one_side_data.time_points, 10)
    traces = decimate(one_side_data.sorted_traces, 10)
    ax1.plot(time_points, mean_trace, lw=1.5, color='#3F888F')

    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace, traces,
                                                                error_bar_method=error_bar_method)
        ax1.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                         facecolor='#7FB5B5', linewidth=0)

    ax1.axvline(0, color='#808080', linewidth=0.5)
    ax1.set_xlim(one_side_data.params.plot_range)
    ax1.set_ylim(y_range)
    ax1.yaxis.set_ticks(np.arange(y_range[0], y_range[1] + 1, 1))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(y_label + 'z-score')
    ax1.set_title(top_label, x=-0.5, y=np.mean(y_range), fontsize=14, rotation="vertical",
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
                    np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=0.5)
    else:
        ax2.scatter(one_side_data.reaction_times,
                    np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=0.5)

    # next trial
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

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(11, 8.25))
    fig.tight_layout(pad=4)
    #fig.tight_layout(pad=2.1, rect=[0,0,0.05,0])  # 2.1)
    font = {'size': 10}
    plt.rc('font', **font)

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

        aligned_data.ipsi_data.params.plot_range = x_range
        aligned_data.contra_data.params.plot_range = x_range

        contra_heatmap = plot_one_side(aligned_data.contra_data, fig, axs[row, 0], axs[row, 1], y_range, hm_range,
                                       error_bar_method=error_bar_method, sort=sort, white_dot=white_dot,
                                       y_label='Contra ', top_label=alignement)
        ipsi_heatmap = plot_one_side(aligned_data.ipsi_data, fig, axs[row, 2], axs[row, 3], y_range, hm_range,
                                     error_bar_method=error_bar_method, sort=sort, white_dot=white_dot,
                                     y_label='Ipsi ', top_label='')
        text = SessionData.mouse + '_' + SessionData.date + '_' + SessionData.recording_site + '_' + \
               SessionData.fiber_side + ', protocol: ' + SessionData.protocol + 'performance: ' + str("%.2f" % SessionData.performance) + '% in n = ' + str("%.0f" % SessionData.nr_trials) + ' trials'

        axs[0, 0].text(x_range[0]-2.8, y_range[1]+0.5, text, fontsize=12)



    return fig


def CueResponses_DMS_vs_TS(all_experiments, mice, locations, main_directory, error_bar_method='sem', x_range=[-2, 3]):

    fig, axs = plt.subplots(nrows=len(mice), ncols=4, figsize=(11, 8.25))
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
                    print('Next:' + mouse +' '+ date +' '+ location)
                    if np.max(data.cue.contra_data.mean_trace) > 0.5 or np.max(data.choice.contra_data.mean_trace) > 0.5:
                        try:
                            if data.performance > 70 and data.protocol == '2AC':
                                print(' > Plotting: ' + mouse + '_' + location + '_' + date)
                                column = locations.index(location)
                                aligned_traces = data.cue.contra_data
                                ax=axs[row, column]
                                ax_overlay=axs[row, 2]
                                plot_avg_sem(aligned_traces, fig, ax, ax_overlay, y_range, error_bar_method='sem', y_label='contra', top_label= mouse + '_' + location,color=color)
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







# function by YJ to analyse a single session of a single mouse

if __name__ == '__main__':
    mice = ['TS16'] #,'TS20']
    dates = ['20230512']
    #plot = 'SingleSession'
    plot = 1

    all_experiments = get_all_experimental_records()
    main_directory = 'Z:\\users\\Yvonne\\photometry_2AC\\'
    if plot == 1: # 'SingleSession':
        print('SingleSession')
        for mouse in mice:
            for date in dates:
                experiment = all_experiments[
                    (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
                fiber_side = experiment['fiber_side'].values[0]
                recording_site = experiment['recording_site'].values[0]
                data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)
                text = mouse + '_' + date + '_' + recording_site + '_' + fiber_side
                figure = heat_map_and_mean_SingleSession(data, error_bar_method='sem', sort=True, x_range=[-2, 3],
                                                    white_dot='default')
                canvas = FigureCanvasSVG(figure)
                canvas.print_svg(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.svg', dpi=600)
                #plt.savefig(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.png', bbox_inches="tight", dpi=600)
                #plt.savefig(main_directory + 'YJ_results\\' + mouse + '\\' + 'Session_' + text + '.pdf', bbox_inches="tight", dpi=600)

                plt.show()

    if plot == 2:
        print(mice)
        locations = ['TS','DMS']
        figure = CueResponses_DMS_vs_TS(all_experiments, mice, locations, main_directory, error_bar_method='sem', x_range=[-2, 3])
        plt.show()

