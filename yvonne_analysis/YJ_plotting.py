import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from tqdm import tqdm
import matplotlib
from scipy.signal import decimate

def make_y_lims_same(ylim_A, ylim_B):
    ylim_min = min(ylim_A[0], ylim_B[0])
    ylim_max = max(ylim_A[1], ylim_B[1])
    return ylim_min, ylim_max

def plot_one_side(one_side_data, fig,  ax1, ax2, dff_range=None, error_bar_method='sem', sort=False, white_dot='default'):
    mean_trace = decimate(one_side_data.mean_trace, 10)
    time_points = decimate(one_side_data.time_points, 10)
    traces = decimate(one_side_data.sorted_traces, 10)
    ax1.plot(time_points, mean_trace, lw=1.5, color='#3F888F')

    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                traces,
                                                                error_bar_method=error_bar_method)
        ax1.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                            facecolor='#7FB5B5', linewidth=0)


    ax1.axvline(0, color='k', linewidth=1)
    ax1.set_xlim(one_side_data.params.plot_range)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('z-score')

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

    heat_im = ax2.imshow(one_side_data.sorted_traces, aspect='auto',
                            extent=[-10, 10, one_side_data.sorted_traces.shape[0], 0], cmap='viridis')

    ax2.axvline(0, color='w', linewidth=1)
    if white_dot == 'reward':
        ax2.scatter(one_side_data.outcome_times,
                       np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=1)
    else:
        ax2.scatter(one_side_data.reaction_times,
                    np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=1)
    ax2.scatter(one_side_data.sorted_next_poke,
                   np.arange(one_side_data.sorted_next_poke.shape[0]) + 0.5, color='k', s=1)
    ax2.tick_params(labelsize=10)
    ax2.set_xlim(one_side_data.params.plot_range)
    ax2.set_ylim([one_side_data.sorted_traces.shape[0], 0])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Trial (sorted)')
    if dff_range:
        vmin = dff_range[0]
        vmax = dff_range[1]
        edge = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        heat_im.set_norm(norm)
    return heat_im

def heat_map_and_mean(aligned_session_data, *mean_data, error_bar_method='sem', sort=False, mean_across_mice=False, xlims=[-2, 2], white_dot='default'):
    if mean_across_mice:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 4))
        fig.tight_layout(pad=1.3)
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5.5, 5.5))
        fig.tight_layout(pad=2.1)

    font = {'size': 10}
    matplotlib.rc('font', **font)

    min_dff_ipsi = np.min(aligned_session_data.ipsi_data.sorted_traces)
    max_dff_ipsi = np.max(aligned_session_data.ipsi_data.sorted_traces)
    min_dff_contra = np.min(aligned_session_data.contra_data.sorted_traces)
    max_dff_contra = np.max(aligned_session_data.contra_data.sorted_traces)
    heatmap_min, heatmap_max = make_y_lims_same((min_dff_ipsi, max_dff_ipsi), (min_dff_contra, max_dff_contra))
    dff_range = (heatmap_min, heatmap_max)

    ipsi_heatmap = plot_one_side(aligned_session_data.ipsi_data, fig, axs[1, 0], axs[1, 1], dff_range,
                                 error_bar_method=error_bar_method, sort=sort, white_dot=white_dot)
    contra_heatmap = plot_one_side(aligned_session_data.contra_data, fig, axs[0, 0], axs[0, 1], dff_range,
                                   error_bar_method=error_bar_method, sort=sort, white_dot=white_dot)
    ylim_ipsi = axs[1, 0].get_ylim()
    ylim_contra = axs[0, 0].get_ylim()
    ylim_min, ylim_max = make_y_lims_same(ylim_ipsi, ylim_contra)
    axs[0, 0].set_ylim([ylim_min, ylim_max])
    axs[1, 0].set_ylim([ylim_min, ylim_max])
    axs[0, 0].set_xlim(xlims)
    axs[1, 0].set_xlim(xlims)
    axs[1, 1].set_xlim(xlims)
    axs[0, 1].set_xlim(xlims)
    axs[0, 0].set_ylabel('z-score')
    axs[1, 0].set_ylabel('z-score')

    cb_ipsi = fig.colorbar(ipsi_heatmap, ax=axs[1, 1], orientation='vertical', fraction=.1)
    cb_contra = fig.colorbar(contra_heatmap, ax=axs[0, 1], orientation='vertical', fraction=.1)
    cb_ipsi.ax.set_title('z-score', fontsize=9, pad=2)
    cb_contra.ax.set_title('z-score', fontsize=9, pad=2)

    if mean_across_mice:
        x_range = axs[0, 0].get_xlim()
        ipsi_data = mean_data[0]
        contra_data = mean_data[1]
        line_plot_dff(aligned_session_data.ipsi_data.time_points, ipsi_data, axs[1, 2], x_range)
        line_plot_dff(aligned_session_data.ipsi_data.time_points, contra_data, axs[0, 2], x_range)
        ylim_ipsi = axs[1, 2].get_ylim()
        ylim_contra = axs[0, 2].get_ylim()
        ylim_min, ylim_max = make_y_lims_same(ylim_ipsi, ylim_contra)
        axs[0, 2].set_ylim([ylim_min, ylim_max])
        axs[1, 2].set_ylim([ylim_min, ylim_max])

        for ax in [axs[0, 0], axs[1, 0]]:
            adjust_label_distances(ax, x_space=0.2, y_space=0.12)
        for ax in [axs[0, 1], axs[1, 1], axs[0, 2], axs[1, 2]]:
            adjust_label_distances(ax, x_space=0.2, y_space=0.2)

    return fig