import matplotlib

from yvonne_basic_functions import *
from YJ_plotting import *
from scipy.signal import decimate
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# If you’re exporting text, you need to make sure matplotlib is exporting editable text, otherwise Illustrator will treat every single character as a shape instead of text. By default matplotlib exports “Type 3 fonts” which Adobe Illustrator doesn’t understand, so you need to change matplotlib to export Type 2/TrueType fonts.
# This setting is, for some reason, the number 42. Run this once at the top of your code and you’ll be set for everything else in the script/notebook.


if __name__ == '__main__':

    main_directory = 'Z:\\users\\Yvonne\\photometry_2AC\\'
    all_experiments = get_all_experimental_records()
    plot = 'RTC_group_plot'
    #plot = 'SOR_group_plot'

    if plot == 'RTC_group_plot':
        mice = ['TS3','TS20','TS21','TS26','TS29']
        dates = ['20230203','20230512','20230510','20230929','20230927']
        x_range = [-2, 3]
        y_range = [-0.5, 1]
        for i, mouse in enumerate(mice):
            nr_mice = len(mice)
            date = dates[i]
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            # get RTC data:
            search_trial_data = mouse + '_' + dates[i] + '_RTC_restructured_data.pkl'
            search_df_data = mouse + '_' + dates[i] + '_RTC_smoothed_signal.npy'
            files_in_path = os.listdir(trial_data_path)
            trial_data_file = [file for file in files_in_path if search_trial_data in file][0]
            df_data_file = [file for file in files_in_path if search_df_data in file][0]
            RTC_trial_data = pd.read_pickle(trial_data_path + trial_data_file)
            RTC_df = np.load(trial_data_path + df_data_file)
            RTC_data = ZScoredTraces_RTC(RTC_trial_data, RTC_df, x_range)
            # get movement signal from same day session:
            experiment = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
            fiber_side = experiment['fiber_side'].values[0]
            recording_site = experiment['recording_site'].values[0]
            data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)

            #print(mouse + '_' + date + '_' + data.protocol + '_&_Random_Tone_Clouds')


            if data.protocol == 'SOR':
                APE_aligned_data = decimate(data.SOR_choice.contra_data.mean_trace, 10)
                APE_time = decimate(data.SOR_choice.contra_data.time_points, 10)
            else:
                APE_aligned_data = decimate(data.choice.contra_data.mean_trace, 10)
                APE_time = decimate(data.choice.contra_data.time_points,10)
            RTC_aligned_data = decimate(RTC_data.mean_trace, 10)
            RTC_time = decimate(RTC_data.time_points, 10)

            if i == 0:
                APE_traces = np.zeros((nr_mice, len(APE_aligned_data)))
                RTC_traces = np.zeros((nr_mice, len(RTC_aligned_data)))
                APE_sem_traces_upper = np.zeros((nr_mice, len(APE_aligned_data)))
                APE_sem_traces_lower = np.zeros((nr_mice, len(APE_aligned_data)))
                RTC_sem_traces_upper = np.zeros((nr_mice, len(RTC_aligned_data)))
                RTC_sem_traces_lower = np.zeros((nr_mice, len(RTC_aligned_data)))
                APE_peak_values = []
                RTC_peak_values = []



            APE_traces[i,:] = APE_aligned_data
            APE_sem_traces = decimate(data.choice.contra_data.sorted_traces,10)
            APE_sem_traces_lower[i,:], APE_sem_traces_upper[i,:] = calculate_error_bars(APE_aligned_data, APE_sem_traces,
                                                                    error_bar_method='sem')
            RTC_traces[i,:] = RTC_aligned_data
            RTC_sem_traces = decimate(RTC_data.sorted_traces,10)
            RTC_sem_traces_lower[i,:], RTC_sem_traces_upper[i,:] = calculate_error_bars(RTC_aligned_data, RTC_sem_traces,
                                                                    error_bar_method='sem')
            # get the peak values:   # APE_time: 16000 datapoints, half: 8000 datapoints = time 0, only consider time after 0
            start_inx = 8000
            APE_range = APE_aligned_data[start_inx:start_inx+8000]
            APE_time_range = APE_time[start_inx:start_inx+8000]
            RTC_range = RTC_aligned_data[start_inx:start_inx+8000]

            APE_peak_index = np.argmax(APE_range) # from time 0 to 8s
            APE_peak_time = APE_time_range[APE_peak_index]
            APE_peak_value = APE_range[APE_peak_index]
            #print('Index: ' + str(APE_peak_index) + ', value: ' + str(APE_peak_value) + ', time: ' + str(APE_peak_time))
            RTC_peak_value = RTC_range[APE_peak_index]
            APE_peak_values.append(APE_peak_value)
            RTC_peak_values.append(RTC_peak_value)

        # calculate mean and sem across mice:
        APE_mean_trace = np.mean(APE_traces, axis=0)
        RTC_mean_trace = np.mean(RTC_traces, axis=0)
        APE_sem_trace = np.std(APE_traces, axis=0)/np.sqrt(nr_mice)
        RTC_sem_trace = np.std(RTC_traces, axis=0)/np.sqrt(nr_mice)


        # plot:
            #colors = plt.cm.viridis(np.linspace(0, 1, 2))
        fig, ax = plt.subplots(2, 1, figsize=(3, 5))
        ax[0].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
        ax[0].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F')
        ax[0].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace, color='#7FB5B5', linewidth=1, alpha=1)
        ax[1].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
        ax[1].plot(RTC_time, RTC_mean_trace, lw=2, color='#e377c2')
        ax[1].fill_between(RTC_time, RTC_mean_trace - RTC_sem_trace, RTC_mean_trace + RTC_sem_trace, facecolor='#e377c2', linewidth=1, alpha=0.3)
        ax[0].text(2, 0.9, 'n = ' + str(nr_mice) + ' mice', fontsize=7)
        title = ['APE', 'RTC']
        for n, a in enumerate(ax):
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

            a.set_ylim(y_range)
            a.set_ylabel('Z-scored dF/F')
            a.set_xlabel('Time (s)')
            a.set_xlim(x_range)
            a.yaxis.set_ticks([-0.5, 0, 0.5, 1])
            a.set_title('  ' + title[n], fontsize=12, loc='left')
            #a.tick_params(axis='both', which='both', length=0)

        plt.tight_layout()
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot.pdf', dpi=300, transparent=True)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot.png', dpi=300, transparent=True)
        plt.show()

        fig, ax = plt.subplots(2, 1+ nr_mice, figsize=(3 + 3*nr_mice, 5))

        y_range = [-0.5, 1.5]
        for i in range(0, nr_mice + 1): # columns, 1 per mouse plus the summary column in the end
            for a in range(0,2): # 2 rows, top row: APE, bottom row: RTC
                ax[a,i].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
                if a == 0:
                    if i <= nr_mice - 1:
                        ax[a, i].plot(APE_time, APE_traces[i,:], lw=2, color='#3F888F')
                        ax[a, i].fill_between(APE_time, APE_sem_traces_lower[i,:], APE_sem_traces_upper[i,:], color='#7FB5B5', linewidth=1, alpha=1)
                    else: # mean trace
                        ax[a, i].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F')
                        ax[a, i].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace, color='#7FB5B5', linewidth=1, alpha=1)
                else:
                    if i <= nr_mice - 1:
                        ax[a, i].plot(RTC_time, RTC_traces[i,:], lw=2, color='#e377c2')
                        ax[a, i].fill_between(RTC_time, RTC_sem_traces_lower[i,:], RTC_sem_traces_upper[i,:], facecolor='#e377c2', linewidth=1, alpha=0.3)
                    else: # mean trace
                        ax[a, i].plot(RTC_time, RTC_mean_trace, lw=2, color='#e377c2')
                        ax[a, i].fill_between(RTC_time, RTC_mean_trace - RTC_sem_trace, RTC_mean_trace + RTC_sem_trace, color='#e377c2', linewidth=1, alpha=0.3)
                ax[a, i].set_ylim(y_range)
                ax[a, i].set_ylabel('Z-scored dF/F')
                ax[a, i].set_xlabel('Time (s)')
                ax[a, i].set_xlim(x_range)
                ax[a, i].yaxis.set_ticks([-0.5, 0.5, 1.5])
                if a == 0:
                    ax[a, i].set_title('  APE', fontsize=12, loc='left')
                    if i <= nr_mice - 1:
                        ax[a, i].text(2, y_range[1]+0.1, mice[i] + ', ' + dates[i], fontsize=7)
                    else:
                        ax[a, i].text(2, y_range[1]+0.1, 'n = ' + str(nr_mice) + ' mice', fontsize=7)
                else:
                    ax[a, i].set_title('  RTC', fontsize=12, loc='left')

                ax[a, i].spines['top'].set_visible(False)
                ax[a, i].spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_all_mice.pdf', dpi=300, transparent=True)
        #plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_all_mice.png', dpi=300, transparent=True)
        plt.show()

        fig, ax = plt.subplots(1, 1 , figsize=(5, 5))

        for i in range(0,len(APE_peak_values)):
            x_val = [0,1]
            y_val = [APE_peak_values[i], RTC_peak_values[i]]
            ax.plot(x_val, y_val, color='#3F888F', linewidth=0.5, marker = 'o', markersize=5)
            #ax.scatter(x_val, y_val, color='#3F888F', s=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([0, 1], labels=["APE", "RTC"])
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot == 'SOR_group_plot':
        print(plot)
        mice = ['TS24', 'TS26', 'TS27','TS29']
        dates = ['20230929', '20230918','20231003', '20230918']
        x_range = [-2, 3]
        y_range = [-1, 2]

        fig1, ax1 = plt.subplots(3, len(mice)*2, figsize=(6 * len(mice), 10)) # each mouse
        fig2, ax2 = plt.subplots(3, 2, figsize=(6, 10)) # group average

        all_group_data = {'SOR_cue':[], 'SOR_choice':[], 'SOR_reward':[], 'cue':[], 'choice':[], 'reward':[]}
        for i, mouse in enumerate(mice):
            print('     >' + mouse + ' ' + dates[i])
            nr_mice = len(mice)
            date = dates[i]
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            # get SOR data:
            experiment = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
            #print(experiment)
            fiber_side = experiment['fiber_side'].values[0]
            recording_site = experiment['recording_site'].values[0]
            data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)


            #print(vars(data))  # get all attributes of the data object

            alignements = ['SOR_cue', 'SOR_choice', 'SOR_reward', 'cue', 'choice', 'reward']

            for a, alignement in enumerate(alignements):

                if alignement == 'SOR_cue':
                    curr_data = data.SOR_cue
                elif alignement == 'SOR_choice':
                    curr_data = data.SOR_choice
                elif alignement == 'SOR_reward':
                    curr_data = data.SOR_reward
                elif alignement == 'cue':
                    curr_data = data.cue
                elif alignement == 'choice':
                    curr_data = data.choice
                elif alignement == 'reward':
                    curr_data = data.reward

                curr_data_mean = decimate(curr_data.contra_data.mean_trace, 10)
                curr_data_time = decimate(curr_data.contra_data.time_points, 10)
                curr_data_traces = decimate(curr_data.contra_data.sorted_traces, 10)

                curr_data_sem_upper = np.zeros(curr_data_mean.shape)
                curr_data_sem_lower = np.zeros(curr_data_mean.shape)
                curr_data_sem_lower, curr_data_sem_upper = calculate_error_bars(curr_data_mean,
                                                                                            curr_data_traces,
                                                                                            error_bar_method='sem')

                # plot single mouse:
                legend = []
                if a < 3:
                    c = 2*i
                    r=a
                    ax1[r, c].set_title(mouse + '_' + date, fontsize=10, loc='left')
                    legend.append(alignement)
                    color_mean = '#e377c2'
                    color_sem = '#e377c2'
                    alpha = 0.3
                else:
                    c = 2*i + 1
                    r = a -3
                    legend.append('2AC_' + alignement)
                    color_mean = '#3F888F'
                    color_sem = '#7FB5B5'
                    alpha = 1


                ax1[r, c].plot(curr_data_time, curr_data_mean, lw=2, color=color_mean)
                ax1[r, c].fill_between(curr_data_time, curr_data_sem_lower, curr_data_sem_upper, color=color_sem,
                                      linewidth=1, alpha=alpha)
                ax1[r, c].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
                ax1[r, c].set_ylim(y_range)
                ax1[r, c].set_ylabel('Z-scored dF/F')
                ax1[r, c].set_yticks(np.arange(y_range[0], y_range[1] + 1, 1))
                ax1[r, c].set_xlabel('Time (s)')
                ax1[r, c].set_xlim(x_range)
                ax1[r, c].legend(legend, loc='upper right', fontsize=8, frameon=False)
                ax1[r, c].spines['top'].set_visible(False)
                ax1[r, c].spines['right'].set_visible(False)
                fig1.tight_layout(pad=4)

                all_group_data[alignement].append(curr_data_mean)
                #print(type(all_group_data))

        fig2.tight_layout(pad=4)
        a = 0
        align_to = ['cue', 'choice', 'reward']
        for a, key in enumerate(all_group_data.keys()):
            curr_data = all_group_data[key]
            # len(curr_data) = nr_mice
            # curr_data[0] = data points (after decimate! > 16000)

            curr_data_set_mean = np.mean(curr_data, axis=0)
            curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)
            #print(str(a) + ' ' + key) #' shape of data: ' + str(curr_data.shape))

            legend = []
            if a < 3:
                c = 0
                r = a
                legend.append('SOR')
                color_mean = '#e377c2'
                color_sem = '#e377c2'
                alpha = 0.3
                curr_align = align_to[a]
                ax2[r, c].set_title(curr_align, x=-0.4, y=0.6 * np.mean(y_range), fontsize=14, rotation="vertical",
                              color='#3F888F')
            else:
                c = 1
                r = a - 3
                legend.append('2AC') # ('2AC_' + key)
                color_mean = '#3F888F'
                color_sem = '#7FB5B5'
                alpha = 1


            ax2[0, 1].set_title('n = ' + str(nr_mice) + ' mice', fontsize=12, loc='right')
            ax2[r, c].plot(curr_data_time, curr_data_set_mean, lw=2, color=color_mean)
            ax2[r, c].fill_between(curr_data_time, curr_data_set_mean - curr_data_set_sem, curr_data_set_mean + curr_data_set_sem, color=color_sem,
                                   linewidth=1, alpha=alpha)
            if r == 0:
                ax2[r, c].legend(legend, loc='lower right', fontsize=8, frameon=False)
            ax2[r, c].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
            ax2[r, c].spines['top'].set_visible(False)
            ax2[r, c].spines['right'].set_visible(False)
            ax2[r, c].set_ylim(y_range)
            ax2[r, c].set_ylabel('Z-scored dF/F')
            ax2[r, c].set_yticks(np.arange(y_range[0], y_range[1] + 1, 1))
            ax2[r, c].set_xlabel('Time (s)')
            ax2[r, c].set_xlim(x_range)

        #plt.tight_layout()
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + '2AC_vs_SOR_group_plot_all_mice.pdf', dpi=300, transparent=True)
        #plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_all_mice.png', dpi=300, transparent=True)

        plt.show()


