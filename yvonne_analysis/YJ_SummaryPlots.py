import matplotlib
import numpy as np

from yvonne_basic_functions import *
from YJ_plotting import *
from scipy.signal import decimate
from scipy.stats import f_oneway
from scipy.stats import levene
from scipy.stats import shapiro
from scipy.stats import kruskal
from scipy.stats import ttest_rel
from scipy.stats import ttest_1samp
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# If you’re exporting text, you need to make sure matplotlib is exporting editable text, otherwise Illustrator will treat every single character as a shape instead of text. By default matplotlib exports “Type 3 fonts” which Adobe Illustrator doesn’t understand, so you need to change matplotlib to export Type 2/TrueType fonts.
# This setting is, for some reason, the number 42. Run this once at the top of your code and you’ll be set for everything else in the script/notebook.


if __name__ == '__main__':

    main_directory = 'Z:\\users\\Yvonne\\photometry_2AC\\'
    all_experiments = get_all_experimental_records()
    #plot = 'RTC_group_plot'     # CORRECT FOR PAPER
    #plot = 'SOR_group_plot'
    plot = 'SOR_group_plot_return'
    #plot = 'APE_group_plot'
    #plot = 'RWN_group_plot'
    #plot = 'SOR_group_plot_NatRev'
    #plot = 'RTC_group_plot_NatRev'
    #plot = 'RWN_group_plot_NatRev'

    if plot == 'RTC_group_plot':
        mice = ['TS3','TS20','TS21','TS26','TS33'] # 'TS29_20230927', TS34_20231102
        dates = ['20230203','20230512','20230510','20230929','20231106']

        x_range = [-2, 3]
        y_range = [-0.7, 1.5]
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

            print(mouse + '_' + date + '_' + data.protocol + '_&_Random_Tone_Clouds. Performance: ' + str(data.performance) + ', fiber side: ' + fiber_side + ', recording site: ' + recording_site)


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

        #y_range = [-0.5, 1.5]
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

        #dotplot
        fig, ax = plt.subplots(1, 1 , figsize=(2, 3)) # width, height
        print('APE peak values: ' + str(APE_peak_values))
        print('RTC peak values: ' + str(RTC_peak_values))

        mean_peak_values = [np.mean(APE_peak_values), np.mean(RTC_peak_values)]
        sem_peak_values = [np.std(APE_peak_values)/np.sqrt(len(APE_peak_values)), np.std(RTC_peak_values)/np.sqrt(len(RTC_peak_values))]

        # effect size:
        cohen_d_paired(APE_peak_values, RTC_peak_values)

        # calculate p-value:

        # Shapiro-Wilk test for normality
        stat, p1 = shapiro(APE_peak_values)
        print('Shapiro-Wilk test for normality APE peak values: Statistics=%.3f, p=%.3f' % (stat, p1)) # p > 0.05, normal distribution
        stat, p2 = shapiro(RTC_peak_values)
        print('Shapiro-Wilk test for normality RTC peak values: Statistics=%.3f, p=%.3f' % (stat, p2)) # p > 0.05, normal distribution

        if p1 > 0.05 and p2 > 0.05:
            # Paired t-test
            stat, p = ttest_rel(APE_peak_values, RTC_peak_values)
            print('Paired t-test for normally distributed data: Statistics=%.3f, p=%.4f' % (stat, p))  # p < 0.05, reject null hypothesis, significant difference between groups
        else:
            # Kruskal-Wallis H-test
            stat, p = kruskal(APE_peak_values, RTC_peak_values)
            print('Kruskal-Wallis H-test: Statistics=%.3f, p=%.3f' % (stat, p))  # p < 0.05, reject null hypothesis, significant difference between groups

        # Test if the mean of the data is significantly different from 0
        stat, p = ttest_1samp(APE_peak_values, popmean=0)
        print("APE_peak_values different from 0? t-statistic:", stat, "p-value:", p)
        stat, p = ttest_1samp(RTC_peak_values, popmean=0)
        print("RTC_peak_values different from 0? t-statistic:", stat, "p-value:", p)

        # Levene test for homogeneity of variances
        #stat, p = levene(APE_peak_values, RTC_peak_values)
        #print('Levene test for homogeneity of variances: Statistics=%.3f, p=%.3f' % (stat, p)) # p > 0.05, variances are equal









        for i in range(0,len(APE_peak_values)):
            x_val = [0,1]
            y_val = [APE_peak_values[i], RTC_peak_values[i]]
            ax.plot(x_val, y_val, color='#3F888F', linewidth=0.5, marker = 'o', markersize=10)
            #ax.scatter(x_val, y_val, color='#3F888F', s=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.set_xticks([0, 1], labels=["APE", "RTC"])

        ax.plot(x_val, mean_peak_values, color='r', linewidth=1, marker = 'o', markersize=10)
        ax.plot([0, 0], [mean_peak_values[0] + sem_peak_values[0], mean_peak_values[0] - sem_peak_values[0]], color='r', linewidth=1)
        ax.plot([1, 1], [mean_peak_values[1] + sem_peak_values[1], mean_peak_values[1] - sem_peak_values[1]], color='r', linewidth=1)

        ax.set_xticks([0, 1])
        ax.set_ylabel('Z-scored dF/F')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.5, 1.5)
        ax.yaxis.set_ticks([-0.5, 0.5, 1.5])
        fig.tight_layout(pad=2)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_all_mice_dotplot.pdf', dpi=300,
                    transparent=True)
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    if plot == 'RTC_group_plot_NatRev':
        mice = ['TS3','TS20','TS26'] #,'TS33']
        dates = ['20230203','20230512','20230929'] # ,'20231106']

        # EXCLUDE: TS33_20231106 because SOR session; TS21_20230510 because all non-silent trials are incorrect


        x_range = [-2, 3]
        y_range = [-0.7, 1.5]
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

            print(mouse + '_' + date + '_' + data.protocol + '_&_Random_Tone_Clouds. Performance: ' + str(data.performance))


            if data.protocol == 'SOR':
                APE_aligned_data = decimate(data.SOR_choice.contra_data.mean_trace, 10)
                APE_time = decimate(data.SOR_choice.contra_data.time_points, 10)
            else:
                APE_aligned_data = decimate(data.choice.contra_data.correct.mean_trace, 10)
                APE_time = decimate(data.choice.contra_data.correct.time_points,10)
                xAPE_aligned_data = decimate(data.choice.contra_data.incorrect.mean_trace, 10)
                xAPE_time = decimate(data.choice.contra_data.incorrect.time_points,10)



            RTC_aligned_data = decimate(RTC_data.mean_trace, 10)
            RTC_time = decimate(RTC_data.time_points, 10)

            if i == 0:
                APE_traces = np.zeros((nr_mice, len(APE_aligned_data)))
                xAPE_traces = np.zeros((nr_mice, len(xAPE_aligned_data)))
                RTC_traces = np.zeros((nr_mice, len(RTC_aligned_data)))
                APE_sem_traces_upper = np.zeros((nr_mice, len(APE_aligned_data)))
                APE_sem_traces_lower = np.zeros((nr_mice, len(APE_aligned_data)))
                xAPE_sem_traces_upper = np.zeros((nr_mice, len(xAPE_aligned_data)))
                xAPE_sem_traces_lower = np.zeros((nr_mice, len(xAPE_aligned_data)))
                RTC_sem_traces_upper = np.zeros((nr_mice, len(RTC_aligned_data)))
                RTC_sem_traces_lower = np.zeros((nr_mice, len(RTC_aligned_data)))
                APE_peak_values = []
                xAPE_peak_values = []
                RTC_peak_values = []



            APE_traces[i,:] = APE_aligned_data
            APE_sem_traces = decimate(data.choice.contra_data.correct.sorted_traces,10)
            APE_sem_traces_lower[i,:], APE_sem_traces_upper[i,:] = calculate_error_bars(APE_aligned_data, APE_sem_traces,
                                                                    error_bar_method='sem')
            xAPE_traces[i,:] = xAPE_aligned_data
            xAPE_sem_traces = decimate(data.choice.contra_data.incorrect.sorted_traces,10)
            xAPE_sem_traces_lower[i, :], xAPE_sem_traces_upper[i, :] = calculate_error_bars(xAPE_aligned_data,
                                                                                          xAPE_sem_traces,
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

            # modification for finding the peak within the RTC trace:
            APE_range = APE_aligned_data[start_inx:start_inx + 2000]
            APE_peak_index = np.argmax(APE_range)  # from time 0 to 2s
            RTC_peak_value = RTC_range[APE_peak_index]

            APE_peak_values.append(APE_peak_value)
            RTC_peak_values.append(RTC_peak_value)

        # calculate mean and sem across mice:
        APE_mean_trace = np.mean(APE_traces, axis=0)
        xAPE_mean_trace = np.mean(xAPE_traces, axis=0)
        RTC_mean_trace = np.mean(RTC_traces, axis=0)
        APE_sem_trace = np.std(APE_traces, axis=0)/np.sqrt(nr_mice)
        xAPE_sem_trace = np.std(xAPE_traces, axis=0) / np.sqrt(nr_mice)
        RTC_sem_trace = np.std(RTC_traces, axis=0)/np.sqrt(nr_mice)


        # plot:
            #colors = plt.cm.viridis(np.linspace(0, 1, 2))
        fig, ax = plt.subplots(2, 1, figsize=(3, 5))
        ax[0].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
        ax[0].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F')
        ax[0].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace, color='#7FB5B5',
                           linewidth=1, alpha=1)
        ax[0].plot(xAPE_time, xAPE_mean_trace, lw=2, color='red')
        ax[0].fill_between(xAPE_time, xAPE_mean_trace - xAPE_sem_trace, xAPE_mean_trace + xAPE_sem_trace, color='red',
                           linewidth=1, alpha=0.4)

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
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_NatRev.pdf', dpi=300, transparent=True)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_NatRev.png', dpi=300, transparent=True)
        plt.show()


        # Plot individual mice:
        fig, ax = plt.subplots(2, 1+ nr_mice, figsize=(3 + 3*nr_mice, 5))

        #y_range = [-0.5, 1.5]
        for i in range(0, nr_mice + 1): # columns, 1 per mouse plus the summary column in the end
            for a in range(0,2): # 2 rows, top row: APE, bottom row: RTC
                ax[a,i].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
                if a == 0:
                    if i <= nr_mice - 1:
                        ax[a, i].plot(APE_time, APE_traces[i,:], lw=2, color='#3F888F')
                        ax[a, i].fill_between(APE_time, APE_sem_traces_lower[i,:], APE_sem_traces_upper[i,:], color='#7FB5B5', linewidth=1, alpha=1)
                        ax[a, i].plot(xAPE_time, xAPE_traces[i,:], lw=2, color='red')
                        ax[a, i].fill_between(xAPE_time, xAPE_sem_traces_lower[i,:], xAPE_sem_traces_upper[i,:], color='red', linewidth=1, alpha=0.4)
                    else: # mean trace
                        ax[a, i].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F')
                        ax[a, i].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace, color='#7FB5B5', linewidth=1, alpha=1)
                        ax[a, i].plot(xAPE_time, xAPE_mean_trace, lw=2, color='red')
                        ax[a, i].fill_between(xAPE_time, xAPE_mean_trace - xAPE_sem_trace, xAPE_mean_trace + xAPE_sem_trace, color='red', linewidth=1, alpha=0.4)
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
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_all_mice_NatRev.pdf', dpi=300, transparent=True)
        #plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_all_mice.png', dpi=300, transparent=True)
        plt.show()

        #dotplot
        fig, ax = plt.subplots(1, 1 , figsize=(2, 3)) # width, height
        print('APE peak values: ' + str(APE_peak_values))
        print('RTC peak values: ' + str(RTC_peak_values))

        mean_peak_values = [np.mean(APE_peak_values), np.mean(RTC_peak_values)]
        sem_peak_values = [np.std(APE_peak_values)/np.sqrt(len(APE_peak_values)), np.std(RTC_peak_values)/np.sqrt(len(RTC_peak_values))]
        for i in range(0,len(APE_peak_values)):
            x_val = [0,1]
            y_val = [APE_peak_values[i], RTC_peak_values[i]]
            ax.plot(x_val, y_val, color='#3F888F', linewidth=0.5, marker = 'o', markersize=10)
            #ax.scatter(x_val, y_val, color='#3F888F', s=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.set_xticks([0, 1], labels=["APE", "RTC"])

        ax.plot(x_val, mean_peak_values, color='r', linewidth=1, marker = 'o', markersize=10)
        ax.plot([0, 0], [mean_peak_values[0] + sem_peak_values[0], mean_peak_values[0] - sem_peak_values[0]], color='r', linewidth=1)
        ax.plot([1, 1], [mean_peak_values[1] + sem_peak_values[1], mean_peak_values[1] - sem_peak_values[1]], color='r', linewidth=1)

        ax.set_xticks([0, 1])
        ax.set_ylabel('Z-scored dF/F')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.5, 1.5)
        ax.yaxis.set_ticks([-0.5, 0.5, 1.5])
        fig.tight_layout(pad=2)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_all_mice_dotplot_NatRev.pdf', dpi=300,
                    transparent=True)
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot == 'RWN_group_plot':
        # 'TS29_20230927', TS34_20231102


        # First WN session, meaningless because no photometry signal, on request of marcus:
        #mice = ['TS32','TS33','TS34']
        #dates = ['20231117', '20231116','20231117']

        # Cell paper:
        mice = ['TS32','TS33','TS34']
        dates = ['20231128','20231128','20231128']

        x_range = [-2, 3]
        y_range = [-1, 1]
        performances = []
        trial_numbers = []
        for i, mouse in enumerate(mice):
            print(mouse)
            nr_mice = len(mice)
            date = dates[i]
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            # get RWN data:
            search_trial_data = mouse + '_' + dates[i] + '_RWN_restructured_data.pkl'
            search_df_data = mouse + '_' + dates[i] + '_RWN_smoothed_signal.npy'
            files_in_path = os.listdir(trial_data_path)
            trial_data_file = [file for file in files_in_path if search_trial_data in file][0]
            df_data_file = [file for file in files_in_path if search_df_data in file][0]
            RWN_trial_data = pd.read_pickle(trial_data_path + trial_data_file)
            RWN_df = np.load(trial_data_path + df_data_file)
            RWN_data = ZScoredTraces_RTC(RWN_trial_data, RWN_df, x_range)
            trial_numbers.append(RWN_data.events_of_int.shape[0])
            # get movement signal from same day session:
            experiment = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
            fiber_side = experiment['fiber_side'].values[0]
            recording_site = experiment['recording_site'].values[0]
            data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)
            performances.append(data.performance)

            print(mouse + '_' + fiber_side + '_' + date + '_' + data.protocol + '_&_Random_WhiteNoise. Performance: ' + str("%.2f" % (data.performance)) + "%, Trial # " + str("%.2f" % (RWN_data.events_of_int.shape[0])))


            if data.protocol == 'SOR':
                APE_aligned_data = decimate(data.SOR_choice.contra_data.mean_trace, 10)
                APE_time = decimate(data.SOR_choice.contra_data.time_points, 10)
            else:
                APE_aligned_data = decimate(data.choice.contra_data.mean_trace, 10)
                APE_time = decimate(data.choice.contra_data.time_points,10)
            RWN_aligned_data = decimate(RWN_data.mean_trace, 10)
            RWN_time = decimate(RWN_data.time_points, 10)

            if i == 0:
                APE_traces = np.zeros((nr_mice, len(APE_aligned_data)))
                RWN_traces = np.zeros((nr_mice, len(RWN_aligned_data)))
                APE_sem_traces_upper = np.zeros((nr_mice, len(APE_aligned_data)))
                APE_sem_traces_lower = np.zeros((nr_mice, len(APE_aligned_data)))
                RWN_sem_traces_upper = np.zeros((nr_mice, len(RWN_aligned_data)))
                RWN_sem_traces_lower = np.zeros((nr_mice, len(RWN_aligned_data)))
                APE_peak_values = []
                RWN_peak_values = []



            APE_traces[i,:] = APE_aligned_data
            APE_sem_traces = decimate(data.choice.contra_data.sorted_traces,10)
            APE_sem_traces_lower[i,:], APE_sem_traces_upper[i,:] = calculate_error_bars(APE_aligned_data, APE_sem_traces,
                                                                    error_bar_method='sem')
            RWN_traces[i,:] = RWN_aligned_data
            RWN_sem_traces = decimate(RWN_data.sorted_traces,10)
            RWN_sem_traces_lower[i,:], RWN_sem_traces_upper[i,:] = calculate_error_bars(RWN_aligned_data, RWN_sem_traces,
                                                                    error_bar_method='sem')
            # get the peak values:   # APE_time: 16000 datapoints, half: 8000 datapoints = time 0, only consider time after 0
            start_inx = 8000
            APE_range = APE_aligned_data[start_inx:start_inx+8000]
            APE_time_range = APE_time[start_inx:start_inx+8000]
            RWN_range = RWN_aligned_data[start_inx:start_inx+8000]

            APE_peak_index = np.argmax(APE_range) # from time 0 to 8s
            APE_peak_time = APE_time_range[APE_peak_index]
            APE_peak_value = APE_range[APE_peak_index]

            #print("peak index of APE: " + str(APE_peak_index))

            if APE_peak_index > len(RWN_range): # if there is no APE peak, the max could be really late in the trace, outside of the RWN_range
                RWN_peak_value = RWN_range.mean()
            else: # this is the normal case, APE peaks relatively soon, within the range of RWN data points
                RWN_peak_value = RWN_range[APE_peak_index]

            #print('Index: ' + str(APE_peak_index) + ', value: ' + str(APE_peak_value) + ', time: ' + str(APE_peak_time))

            APE_peak_values.append(APE_peak_value)
            RWN_peak_values.append(RWN_peak_value)

        # calculate mean and sem across mice:
        APE_mean_trace = np.mean(APE_traces, axis=0)
        RWN_mean_trace = np.mean(RWN_traces, axis=0)
        APE_sem_trace = np.std(APE_traces, axis=0)/np.sqrt(nr_mice)
        RWN_sem_trace = np.std(RWN_traces, axis=0)/np.sqrt(nr_mice)

        print('trial numbers: ' + str("%.2f" % np.mean(trial_numbers)) + ' +/- ' + str("%.2f" % (np.std(trial_numbers) / np.sqrt(nr_mice))))
        print('performance: ' + str("%.2f" % np.mean(performances)) + ' +/- ' + str("%.2f" % (np.std(performances) / np.sqrt(nr_mice))))

        # plot:
            #colors = plt.cm.viridis(np.linspace(0, 1, 2))
        fig, ax = plt.subplots(2, 1, figsize=(3, 5))
        ax[0].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
        ax[0].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F')
        ax[0].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace, color='#7FB5B5', linewidth=1, alpha=1)
        ax[1].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
        ax[1].plot(RWN_time, RWN_mean_trace, lw=2, color='#e377c2')
        ax[1].fill_between(RWN_time, RWN_mean_trace - RWN_sem_trace, RWN_mean_trace + RWN_sem_trace, facecolor='#e377c2', linewidth=1, alpha=0.3)
        ax[0].text(2, 0.9, 'n = ' + str(nr_mice) + ' mice', fontsize=7)
        title = ['APE', 'RWN']
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
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RWN_group_plot.pdf', dpi=300, transparent=True)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RWN_group_plot.png', dpi=300, transparent=True)
        plt.show()

        # dotplot
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))  # width, height
        print('APE peak values: ' + str(APE_peak_values))
        print('RWN peak values: ' + str(RWN_peak_values))

        mean_peak_values = [np.mean(APE_peak_values), np.mean(RWN_peak_values)]
        sem_peak_values = [np.std(APE_peak_values) / np.sqrt(len(APE_peak_values)),
                           np.std(RWN_peak_values) / np.sqrt(len(RWN_peak_values))]


        # STATISTICS:
        # Effect size:
        # indpendent samples: effect_size = (np.mean(APE_peak_values) - np.mean(RWN_peak_values)) / np.sqrt((np.std(APE_peak_values) ** 2 + np.std(RWN_peak_values) ** 2) / 2)
        # print('Effect size: ' + str(effect_size) + ' (pooled STD)')

        cohen_d_paired(APE_peak_values, RWN_peak_values)


        # calculate p-value:

        # Shapiro-Wilk test for normality
        stat, p1 = shapiro(APE_peak_values)
        print('Shapiro-Wilk test for normality APE peak values: Statistics=%.3f, p=%.3f' % (stat, p1)) # p > 0.05, normal distribution
        stat, p2 = shapiro(RWN_peak_values)
        print('Shapiro-Wilk test for normality RTC peak values: Statistics=%.3f, p=%.3f' % (stat, p2)) # p > 0.05, normal distribution


        if p1 > 0.05 and p2 > 0.05:
            # Paired t-test
            stat, p = ttest_rel(APE_peak_values, RWN_peak_values)
            print('Paired t-test for normally distributed data: Statistics=%.3f, p=%.5f' % (stat, p))  # p < 0.05, reject null hypothesis, significant difference between groups
        else:
            # Kruskal-Wallis H-test
            stat, p = kruskal(APE_peak_values, RWN_peak_values)
            print('Kruskal-Wallis H-test: Statistics=%.3f, p=%.3f' % (stat, p))  # p < 0.05, reject null hypothesis, significant difference between groups

        # Test if the mean of the data is significantly different from 0
        stat, p = ttest_1samp(APE_peak_values, popmean=0)
        print("APE_peak_values different from 0? t-statistic:", stat, "p-value:", p)

        stat, p = ttest_1samp(RWN_peak_values, popmean=0)
        print("RWN_peak_values different from 0? t-statistic:", stat, "p-value:", p)




        for i in range(0, len(APE_peak_values)):
            x_val = [0, 1]
            y_val = [APE_peak_values[i], RWN_peak_values[i]]
            ax.plot(x_val, y_val, color='#3F888F', linewidth=0.5, marker='o', markersize=10)
            # ax.scatter(x_val, y_val, color='#3F888F', s=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.set_xticks([0, 1], labels=["APE", "RTC"])

        #ax.plot(x_val, mean_peak_values, color='r', linewidth=1, marker='o', markersize=10)
        #ax.plot([0, 0], [mean_peak_values[0] + sem_peak_values[0], mean_peak_values[0] - sem_peak_values[0]], color='r',
        #       linewidth=1)
        #ax.plot([1, 1], [mean_peak_values[1] + sem_peak_values[1], mean_peak_values[1] - sem_peak_values[1]], color='r',
        #        linewidth=1)

        ax.set_xticks([0, 1])
        ax.set_ylabel('Z-scored dF/F')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.5,1.5)
        #ax.yaxis.set_ticks([-1, 0, 1])
        fig.tight_layout(pad=2)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RWN_group_plot_all_mice_dotplot.pdf', dpi=300,
                    transparent=True)
        plt.show()


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot == 'RWN_group_plot_NatRev':
        print(plot)
        mice = ['TS32', 'TS33', 'TS34']
        dates = ['20231128', '20231128', '20231128']

        x_range = [-2, 3]
        y_range = [-1, 1]
        performances = []
        trial_numbers = []
        for i, mouse in enumerate(mice):
            print(mouse)
            nr_mice = len(mice)
            date = dates[i]
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            # get RWN data:
            search_trial_data = mouse + '_' + dates[i] + '_RWN_restructured_data.pkl'
            search_df_data = mouse + '_' + dates[i] + '_RWN_smoothed_signal.npy'
            files_in_path = os.listdir(trial_data_path)
            trial_data_file = [file for file in files_in_path if search_trial_data in file][0]
            df_data_file = [file for file in files_in_path if search_df_data in file][0]
            RWN_trial_data = pd.read_pickle(trial_data_path + trial_data_file)
            RWN_df = np.load(trial_data_path + df_data_file)
            RWN_data = ZScoredTraces_RTC(RWN_trial_data, RWN_df, x_range)
            trial_numbers.append(RWN_data.events_of_int.shape[0])
            # get movement signal from same day session:
            experiment = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
            fiber_side = experiment['fiber_side'].values[0]
            recording_site = experiment['recording_site'].values[0]
            data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)
            performances.append(data.performance)

            print(mouse + '_' + date + '_' + data.protocol + '_&_Random_WhiteNoise. Performance: ' + str(
                "%.2f" % (data.performance)) + "%, Trial # " + str("%.2f" % (RWN_data.events_of_int.shape[0])))

            if data.protocol == 'SOR':
                APE_aligned_data = decimate(data.SOR_choice.contra_data.mean_trace, 10)
                APE_time = decimate(data.SOR_choice.contra_data.time_points, 10)
            else:
                APE_aligned_data = decimate(data.choice.contra_data.correct.mean_trace, 10)
                APE_time = decimate(data.choice.contra_data.correct.time_points, 10)
                xAPE_aligned_data = decimate(data.choice.contra_data.incorrect.mean_trace, 10)
                xAPE_time = decimate(data.choice.contra_data.incorrect.time_points, 10)


            RWN_aligned_data = decimate(RWN_data.mean_trace, 10)
            RWN_time = decimate(RWN_data.time_points, 10)

            if i == 0:
                APE_traces = np.zeros((nr_mice, len(APE_aligned_data)))
                xAPE_traces = np.zeros((nr_mice, len(xAPE_aligned_data)))
                RWN_traces = np.zeros((nr_mice, len(RWN_aligned_data)))
                APE_sem_traces_upper = np.zeros((nr_mice, len(APE_aligned_data)))
                APE_sem_traces_lower = np.zeros((nr_mice, len(APE_aligned_data)))
                xAPE_sem_traces_upper = np.zeros((nr_mice, len(xAPE_aligned_data)))
                xAPE_sem_traces_lower = np.zeros((nr_mice, len(xAPE_aligned_data)))
                RWN_sem_traces_upper = np.zeros((nr_mice, len(RWN_aligned_data)))
                RWN_sem_traces_lower = np.zeros((nr_mice, len(RWN_aligned_data)))
                APE_peak_values = []
                xAPE_peak_values = []
                RWN_peak_values = []

            xAPE_traces[i, :] = xAPE_aligned_data
            xAPE_sem_traces = decimate(data.choice.contra_data.incorrect.sorted_traces, 10)
            xAPE_sem_traces_lower[i, :], xAPE_sem_traces_upper[i, :] = calculate_error_bars(xAPE_aligned_data,
                                                                                            xAPE_sem_traces,
                                                                                            error_bar_method='sem')

            APE_traces[i, :] = APE_aligned_data
            APE_sem_traces = decimate(data.choice.contra_data.correct.sorted_traces, 10)
            APE_sem_traces_lower[i, :], APE_sem_traces_upper[i, :] = calculate_error_bars(APE_aligned_data,
                                                                                          APE_sem_traces,
                                                                                          error_bar_method='sem')
            RWN_traces[i, :] = RWN_aligned_data
            RWN_sem_traces = decimate(RWN_data.sorted_traces, 10)
            RWN_sem_traces_lower[i, :], RWN_sem_traces_upper[i, :] = calculate_error_bars(RWN_aligned_data,
                                                                                          RWN_sem_traces,
                                                                                          error_bar_method='sem')

            # get the peak values:   # APE_time: 16000 datapoints, half: 8000 datapoints = time 0, only consider time after 0
            start_inx = 8000
            APE_range = APE_aligned_data[start_inx:start_inx + 8000]
            APE_time_range = APE_time[start_inx:start_inx + 8000]
            RWN_range = RWN_aligned_data[start_inx:start_inx + 8000]

            APE_peak_index = np.argmax(APE_range)  # from time 0 to 8s
            APE_peak_time = APE_time_range[APE_peak_index]
            APE_peak_value = APE_range[APE_peak_index]

            # print("peak index of APE: " + str(APE_peak_index))

            if APE_peak_index > len(
                    RWN_range):  # if there is no APE peak, the max could be really late in the trace, outside of the RWN_range
                RWN_peak_value = RWN_range.mean()
            else:  # this is the normal case, APE peaks relatively soon, within the range of RWN data points
                RWN_peak_value = RWN_range[APE_peak_index]

            # print('Index: ' + str(APE_peak_index) + ', value: ' + str(APE_peak_value) + ', time: ' + str(APE_peak_time))

            APE_peak_values.append(APE_peak_value)
            RWN_peak_values.append(RWN_peak_value)

        # calculate mean and sem across mice:
        APE_mean_trace = np.mean(APE_traces, axis=0)
        xAPE_mean_trace = np.mean(xAPE_traces, axis=0)
        RWN_mean_trace = np.mean(RWN_traces, axis=0)
        APE_sem_trace = np.std(APE_traces, axis=0) / np.sqrt(nr_mice)
        xAPE_sem_trace = np.std(xAPE_traces, axis=0) / np.sqrt(nr_mice)
        RWN_sem_trace = np.std(RWN_traces, axis=0) / np.sqrt(nr_mice)

        print('trial numbers: ' + str("%.2f" % np.mean(trial_numbers)) + ' +/- ' + str(
            "%.2f" % (np.std(trial_numbers) / np.sqrt(nr_mice))))
        print('performance: ' + str("%.2f" % np.mean(performances)) + ' +/- ' + str(
            "%.2f" % (np.std(performances) / np.sqrt(nr_mice))))

        # plot:
        # colors = plt.cm.viridis(np.linspace(0, 1, 2))
        fig, ax = plt.subplots(2, 1, figsize=(3, 5))
        ax[0].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
        ax[0].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F')
        ax[0].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace, color='#7FB5B5',
                           linewidth=1, alpha=1)
        ax[0].plot(xAPE_time, xAPE_mean_trace, lw=2, color='red')
        ax[0].fill_between(xAPE_time, xAPE_mean_trace - xAPE_sem_trace, xAPE_mean_trace + xAPE_sem_trace, color='red',
                           linewidth=1, alpha=0.4)
        ax[1].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
        ax[1].plot(RWN_time, RWN_mean_trace, lw=2, color='#e377c2')
        ax[1].fill_between(RWN_time, RWN_mean_trace - RWN_sem_trace, RWN_mean_trace + RWN_sem_trace,
                           facecolor='#e377c2', linewidth=1, alpha=0.3)
        ax[0].text(2, 0.9, 'n = ' + str(nr_mice) + ' mice', fontsize=7)
        title = ['APE', 'RWN']
        for n, a in enumerate(ax):
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

            a.set_ylim(y_range)
            a.set_ylabel('Z-scored dF/F')
            a.set_xlabel('Time (s)')
            a.set_xlim(x_range)
            a.yaxis.set_ticks([-0.5, 0, 0.5, 1])
            a.set_title('  ' + title[n], fontsize=12, loc='left')
            # a.tick_params(axis='both', which='both', length=0)

        plt.tight_layout()
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RWN_group_plot_NatRev.pdf', dpi=300, transparent=True)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RWN_group_plot_NatRev.png', dpi=300, transparent=True)
        plt.show()

        # Plot individual mice:
        fig, ax = plt.subplots(2, 1 + nr_mice, figsize=(3 + 3 * nr_mice, 5))

        # y_range = [-0.5, 1.5]
        for i in range(0, nr_mice + 1):  # columns, 1 per mouse plus the summary column in the end
            for a in range(0, 2):  # 2 rows, top row: APE, bottom row: RWN
                ax[a, i].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
                if a == 0:
                    if i <= nr_mice - 1:
                        ax[a, i].plot(APE_time, APE_traces[i, :], lw=2, color='#3F888F')
                        ax[a, i].fill_between(APE_time, APE_sem_traces_lower[i, :], APE_sem_traces_upper[i, :],
                                              color='#7FB5B5', linewidth=1, alpha=1)
                        ax[a, i].plot(xAPE_time, xAPE_traces[i, :], lw=2, color='red')
                        ax[a, i].fill_between(xAPE_time, xAPE_sem_traces_lower[i, :], xAPE_sem_traces_upper[i, :],
                                              color='red', linewidth=1, alpha=0.4)
                    else:  # mean trace
                        ax[a, i].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F')
                        ax[a, i].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace,
                                              color='#7FB5B5', linewidth=1, alpha=1)
                        ax[a, i].plot(xAPE_time, xAPE_mean_trace, lw=2, color='red')
                        ax[a, i].fill_between(xAPE_time, xAPE_mean_trace - xAPE_sem_trace,
                                              xAPE_mean_trace + xAPE_sem_trace, color='red', linewidth=1, alpha=0.4)
                else:
                    if i <= nr_mice - 1:
                        ax[a, i].plot(RWN_time, RWN_traces[i, :], lw=2, color='#e377c2')
                        ax[a, i].fill_between(RWN_time, RWN_sem_traces_lower[i, :], RWN_sem_traces_upper[i, :],
                                              facecolor='#e377c2', linewidth=1, alpha=0.3)
                    else:  # mean trace
                        ax[a, i].plot(RWN_time, RWN_mean_trace, lw=2, color='#e377c2')
                        ax[a, i].fill_between(RWN_time, RWN_mean_trace - RWN_sem_trace, RWN_mean_trace + RWN_sem_trace,
                                              color='#e377c2', linewidth=1, alpha=0.3)
                ax[a, i].set_ylim(y_range)
                ax[a, i].set_ylabel('Z-scored dF/F')
                ax[a, i].set_xlabel('Time (s)')
                ax[a, i].set_xlim(x_range)
                ax[a, i].yaxis.set_ticks([-0.5, 0.5, 1.5])
                if a == 0:
                    ax[a, i].set_title('  APE', fontsize=12, loc='left')
                    if i <= nr_mice - 1:
                        ax[a, i].text(2, y_range[1] + 0.1, mice[i] + ', ' + dates[i], fontsize=7)
                    else:
                        ax[a, i].text(2, y_range[1] + 0.1, 'n = ' + str(nr_mice) + ' mice', fontsize=7)
                else:
                    ax[a, i].set_title('  RWN', fontsize=12, loc='left')

                ax[a, i].spines['top'].set_visible(False)
                ax[a, i].spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RWN_group_plot_all_mice_NatRev.pdf', dpi=300,
                    transparent=True)

        plt.show()


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    if plot == 'SOR_group_plot_return':
        print(plot)

        # Sessions used in Cell submission and nature figure ED6 so far
        add = 'classic_SOR_fig'
        mice = ['TS24', 'TS26', 'TS27','TS32','TS33','TS34']
        dates = ['20230929', '20230918','20231003', '20231026','20231102','20231031']

        # First session with sound on (novelty):
        #add = 'novelty_fig'
        #mice = ['TS30', 'TS31', 'TS32', 'TS33', 'TS34'] # 2 mice, TS30, TS31 only a maybe first exposure, because maybe run RTC on dayof habituation
        #dates = ['20231004', '20231004', '20231026', '20231026', '20231026']

        #add = '1stExposureCue_fig' # 1st session post habituation, 1st time hearing cue sound, SOR training, NOVELTY FIGURE
        #mice = ['TS32', 'TS33', 'TS34']
        #dates = ['20231026', '20231026', '20231026']

        time_amplitude = 3 # 0.5    # seconds after cue onset
        x_range = [-2, 3]
        y_range = [-1, 2]


        all_group_data = {'SOR_cue': [], 'SOR_choice': [], 'SOR_reward': [], 'cue': [], 'choice': [], 'reward': [],
                      'SOR_return_cueON_contra': [], 'SOR_return_cueOFF_contra': [], 'SOR_return_cueON_ipsi': [], 'SOR_return_cueOFF_ipsi': []}
        performances = []



        APE_choice_peak_values = []
        return_contra_cue_on_values = []
        return_contra_cue_off_values = []
        return_ipsi_cue_on_values = []
        return_ipsi_cue_off_values = []
        percentage_cue_on = [] # percentage of trials with return cue on per mouse

        for i, mouse in enumerate(mice):
            print('     >' + mouse + ' ' + dates[i])
            nr_return_trials_cue_on = 0
            nr_return_trials_cue_off = 0
            nr_mice = len(mice)
            date = dates[i]
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            # get SOR data:
            experiment = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]

            fiber_side = experiment['fiber_side'].values[0]
            recording_site = experiment['recording_site'].values[0]
            data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)
            performances.append(data.performance)

            print()

            alignements = ['SOR_choice', 'SOR_cue', 'SOR_reward', 'cue', 'choice', 'reward', 'SOR_return_cueON_contra',
                           'SOR_return_cueOFF_contra', 'SOR_return_cueON_ipsi', 'SOR_return_cueOFF_ipsi']

            for a, alignement in enumerate(alignements):

                if alignement == 'SOR_cue':
                    curr_data = data.SOR_cue.contra_data
                elif alignement == 'SOR_choice':
                    curr_data = data.SOR_choice.contra_data


                elif alignement == 'SOR_reward':
                    curr_data = data.SOR_reward.contra_data
                elif alignement == 'cue':
                    curr_data = data.cue.contra_data
                elif alignement == 'choice':
                    curr_data = data.choice.contra_data
                    print('choice traces: ' + str(curr_data.sorted_traces.shape))
                elif alignement == 'reward':
                    curr_data = data.reward.contra_data
                elif alignement == 'SOR_return_cueON_contra':
                    curr_data = data.SOR_return_cueON.contra_data
                elif alignement == 'SOR_return_cueOFF_contra':
                    curr_data = data.SOR_return_cueOFF.contra_data
                elif alignement == 'SOR_return_cueON_ipsi':
                    curr_data = data.SOR_return_cueON.ipsi_data
                elif alignement == 'SOR_return_cueOFF_ipsi':
                    curr_data = data.SOR_return_cueOFF.ipsi_data

                curr_data_mean = decimate(curr_data.mean_trace, 10)

                # curr data of return is aligned to side-port-out (hence return movement start, not return cue onset)
                # shifting the data points:
                if alignement == 'SOR_return_cueON_contra' or alignement == 'SOR_return_cueOFF_contra' or alignement == 'SOR_return_cueON_ipsi' or alignement == 'SOR_return_cueOFF_ipsi':
                    t1 = np.pad(curr_data_mean, (0, 20))  # pad with 200 zeros = 20ms with 10Hz sampling, however decimate 10 above, hence only 20 datapoints
                    t2 = t1[20:]
                    curr_data_mean = t2

                curr_data_time = decimate(curr_data.time_points, 10)
                curr_data_traces = decimate(curr_data.sorted_traces, 10)

                curr_data_sem_upper = np.zeros(curr_data_mean.shape)
                curr_data_sem_lower = np.zeros(curr_data_mean.shape)
                curr_data_sem_lower, curr_data_sem_upper = calculate_error_bars(curr_data_mean,
                                                                                            curr_data_traces,
                                                                                            error_bar_method='sem')
                all_group_data[alignement].append(curr_data_mean)

                # get the peak values:
                if alignement == 'choice':
                    print('choice traces:' + str(curr_data_traces.shape))


                    start_inx = 8000  # time 0
                    # SOR_choice_data_range = [start_inx:start_inx + 8000]
                    APE_choice_time_range = curr_data_time[start_inx:start_inx + 8000]
                    APE_choice_peak_index = np.argmax(curr_data_mean[start_inx:start_inx + 8000])
                    APE_choice_peak_time = APE_choice_time_range[APE_choice_peak_index]
                    APE_choice_peak_value = curr_data_mean[start_inx + APE_choice_peak_index]
                    APE_choice_peak_values.append(APE_choice_peak_value)
                    # print(mouse + ' ' + date + ' ' + alignement + ' ' + str(SOR_choice_peak_value) + ' ' + str(SOR_choice_peak_time))
                    print(APE_choice_peak_index)

                # take amplitude X seconds after cue start
                start_inx = 8000  # time 0
                #timepoint_index = (time_amplitude +0.02)* 1000 + start_inx       #8s before and after aligned time point (which is 0). +20ms for cue delay + desired time after conset (0.5s for ex)
                #amplitude = curr_data_mean[int(timepoint_index)]
                if alignement == 'SOR_return_cueON_contra' or alignement == 'SOR_return_cueOFF_contra' or alignement == 'SOR_return_cueON_ipsi' or alignement == 'SOR_return_cueOFF_ipsi':
                    amplitude = curr_data_mean[start_inx + APE_choice_peak_index]
                    if(i == 0):
                        print('             > Amplitude extracted based on time of peak of choice!!')



                if alignement == 'SOR_return_cueON_contra':
                    #print('>        ' + alignement + ' ' + str(amplitude) + ' ' + str(timepoint_index))
                    nr_return_trials_cue_on = nr_return_trials_cue_on + curr_data_traces.shape[0]
                    return_contra_cue_on_values.append(amplitude)
                elif alignement == 'SOR_return_cueOFF_contra':
                    #print('>        ' + alignement + ' ' + str(amplitude) + ' ' + str(timepoint_index))
                    nr_return_trials_cue_off = nr_return_trials_cue_off + curr_data_traces.shape[0]
                    return_contra_cue_off_values.append(amplitude)
                elif alignement == 'SOR_return_cueON_ipsi':
                    #print('>        ' + alignement + ' ' + str(amplitude) + ' ' + str(timepoint_index))
                    nr_return_trials_cue_on = nr_return_trials_cue_on + curr_data_traces.shape[0]
                    return_ipsi_cue_on_values.append(amplitude)
                elif alignement == 'SOR_return_cueOFF_ipsi':
                    #print('>        ' + alignement + ' ' + str(amplitude) + ' ' + str(timepoint_index))
                    nr_return_trials_cue_off = nr_return_trials_cue_off + curr_data_traces.shape[0]
                    return_ipsi_cue_off_values.append(amplitude)

            #print('             > Return trials: ' + str(nr_return_trials_cue_on) + ' cue on, ' + str(nr_return_trials_cue_off) + ' cue off, ' + 'total cue on % : ' + str(nr_return_trials_cue_on / (nr_return_trials_cue_on + nr_return_trials_cue_off)))
            #percentage_cue_on.append(nr_return_trials_cue_on / (nr_return_trials_cue_on + nr_return_trials_cue_off))


        # general info:
        #print('Performance: ' + str(np.mean(performances)) + ' +/- ' + str(np.std(performances) / np.sqrt(nr_mice)))
        #print('Return trials: ' + str(np.mean(percentage_cue_on)) + ' +/- ' + str(np.std(percentage_cue_on) / np.sqrt(nr_mice)))


        fig1, ax1 = plt.subplots(4, 2, figsize=(6, 10*4/3))  # group average
        fig1.tight_layout(pad=4)
        a = 0
        align_to = ['cue', 'choice', 'reward', 'return']
        legend = []
        for a, key in enumerate(all_group_data.keys()):
            curr_data = all_group_data[key]
            curr_data_set_mean = np.mean(curr_data, axis=0)
            curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)


            if a < 3:
                c = 0
                r = a
                legend = ['SOR']
                #legend.append('SOR')
                color_mean = '#e377c2'
                color_sem = '#e377c2'
                alpha = 0.3
                curr_align = align_to[a]
                ax1[r, c].set_title(curr_align, x=-0.4, y=0.6 * np.mean(y_range), fontsize=14, rotation="vertical",
                              color='#3F888F')
            elif a < 6:
                c = 1
                r = a - 3
                legend=['2AC'] # ('2AC_' + key)
                color_mean = '#3F888F'
                color_sem = '#7FB5B5'
                alpha = 1


            elif a is 6: # return contra cue on
                legend = []
                c = 0
                r = 3
                legend.append('contra return, cue on ')
                color_mean = 'blue'
                color_sem = 'blue'
                alpha = 0.3
                curr_align = align_to[3]
                ax1[r, c].set_title(curr_align, x=-0.4, y=0.6 * np.mean(y_range), fontsize=14, rotation="vertical",
                                    color='#3F888F')

            elif a is 7: # return contra cue off
                c = 0
                r = 3
                legend.append('contra return, cue off')
                color_mean = 'cyan'
                color_sem = 'cyan'
                alpha = 0.3

            elif a is 8: # return ipsi cue on
                legend = []
                c = 1
                r = 3
                legend.append('ipsi return, cue on')
                color_mean = 'blue'
                color_sem = 'blue'
                alpha = 0.3

            elif a is 9: # return ipsi cue off
                c = 1
                r = 3
                legend.append('ipsi return, cue off')
                #legend.append('') # empty legend entry
                color_mean = 'cyan'
                color_sem = 'cyan'
                alpha = 0.3




            ax1[0, 1].set_title('n = ' + str(nr_mice) + ' mice', fontsize=12, loc='right')
            ax1[r, c].plot(curr_data_time, curr_data_set_mean, lw=2, color=color_mean)
            ax1[r, c].fill_between(curr_data_time, curr_data_set_mean - curr_data_set_sem, curr_data_set_mean + curr_data_set_sem, color=color_sem,
                                   linewidth=1, alpha=alpha)
            if r == 0:
                ax1[r, c].legend(legend, loc='lower right', fontsize=8, frameon=False)
            if a == 7 or a == 9:
                ax1[r, c].legend(legend, loc='lower right', fontsize=8, frameon=False)

            if a < 6: # only for SOR and 2AC
                ax1[r, c].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
            ax1[r, c].spines['top'].set_visible(False)
            ax1[r, c].spines['right'].set_visible(False)
            ax1[r, c].set_ylim(y_range)
            ax1[r, c].set_ylabel('Z-scored dF/F')
            ax1[r, c].set_yticks(np.arange(y_range[0], y_range[1] + 1, 1))
            ax1[r, c].set_xlabel('Time (s)')
            ax1[r, c].set_xlim(x_range)

        ax1[3, 0].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
        ax1[3, 1].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
        #plt.tight_layout()

        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + '2AC_vs_SOR_group_RETURN_ca_' + add + str(time_amplitude) + '.pdf', dpi=300,
                    transparent=True)



        # mean values dotplot
        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3)) # dotplot

        mean_peak_values = [np.mean(return_contra_cue_on_values), np.mean(return_contra_cue_off_values), np.mean(return_ipsi_cue_on_values), np.mean(return_ipsi_cue_off_values)]
        sem_peak_values = [np.std(return_contra_cue_on_values) / np.sqrt(len(return_contra_cue_on_values)), np.std(return_contra_cue_off_values) / np.sqrt(len(return_contra_cue_off_values)),
                           np.std(return_ipsi_cue_on_values) / np.sqrt(len(return_ipsi_cue_on_values)), np.std(return_ipsi_cue_off_values) / np.sqrt(len(return_ipsi_cue_off_values))]






        for i in range(0, len(return_contra_cue_on_values)):
            x_val = [0, 1, 2, 3]
            y_val = [return_contra_cue_on_values[i], return_contra_cue_off_values[i], return_ipsi_cue_on_values[i], return_ipsi_cue_off_values[i]]
            ax2.plot(x_val, y_val, color='#3F888F', linewidth=0, marker='o', markersize=10)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            #ax2.set_xticks([0, 1, 2, 3], ["contra cue on", "contra cue off", "ipsi cue on", "ipsi cue off"])
            #ax2.set_xticks([0, 1, 2, 3], labels=["contra cue on", "contra cue off", "ipsi cue on", "ipsi cue off"])
            ax2.set_ylabel('Z-scored dF/F')

        for i in range(0, 4):
            ax2.plot([i, i], [mean_peak_values[i] + sem_peak_values[i], mean_peak_values[i] - sem_peak_values[i]], color='r', linewidth=1)
            ax2.plot([i], [mean_peak_values[i]], marker='o', markersize=10, color='r', linewidth=1)

        ax2.set_ylim([-1, 1])
        ax2.set_yticks([-1, 0, 1])
        ax2.set_xlim([-0.5,3.5])
        #ax2.set_xticks([0,1,2,3])
        #ax2.set_xticks([0, 1, 2, 3], ["contra cue on", "contra cue off", "ipsi cue on", "ipsi cue off"])
        fig2.tight_layout(pad=2)


        #print('contra cue on')
        #print(return_contra_cue_on_values)
        #print('contra cue off')
        #print(return_contra_cue_off_values)
        #print('ipsi cue on')
        #print(return_ipsi_cue_on_values)
        #print('ipsi cue off')
        #print(return_ipsi_cue_off_values)


        # equal variances?
        stat, p = levene(return_contra_cue_on_values, return_contra_cue_off_values, return_ipsi_cue_on_values, return_ipsi_cue_off_values)
        #print(f"Levene's test p-value: {p}")
        if p >= 0.05:  # Variances are equal --> one way ANOVA
            print('Levene test: Variances are equal, one way ANOVA appropriate')
        else:
            print('Variances are not equal, Kruskal-Wallis test or Welch ANOVA appropriate')

        # normal distribution?
        # Combine all groups' data into one array for testing residuals
        all_data = return_contra_cue_on_values + return_contra_cue_off_values + return_ipsi_cue_on_values + return_ipsi_cue_off_values
        stat, p = shapiro(all_data)
        if p >= 0.05: # Data is normally distributed --> one way ANOVA
            print(f"Shapiro-Wilk test p-value: {p}")
            print('Data is normally distributed, one way ANOVA appropriate')
        else:
           # print('Data is not normally distributed, Kruskal-Wallis test appropriate:')
            stat, p = kruskal(return_contra_cue_on_values, return_contra_cue_off_values, return_ipsi_cue_on_values, return_ipsi_cue_off_values)
            print(f"Kruskal-Wallis test p-value: {p}")


        if p >= 0.05: # Variances are equal --> one way ANOVA
            F, p = f_oneway(return_contra_cue_on_values, return_contra_cue_off_values, return_ipsi_cue_on_values, return_ipsi_cue_off_values)
            print('One way Anova, F: ' + str(F))
            print('One way Anova, p: ' + str(p))

        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + '2AC_vs_SOR_group_RETURN' + add + str(time_amplitude) + '_DP.pdf', dpi=300, transparent=True)


        # connected dotplot
        fig3, ax3 = plt.subplots(1, 1, figsize=(2, 3))  # dotplot


        contra_mean_peak_values = [np.mean(return_contra_cue_on_values), np.mean(return_contra_cue_off_values)]
        ipsi_mean_peak_values = [np.mean(return_ipsi_cue_on_values), np.mean(return_ipsi_cue_off_values)]

        contra_sem_peak_values = [np.std(return_contra_cue_on_values) / np.sqrt(len(return_contra_cue_on_values)),
                            np.std(return_contra_cue_off_values) / np.sqrt(len(return_contra_cue_off_values))]
        ipsi_sem_peak_values = [np.std(return_ipsi_cue_on_values) / np.sqrt(len(return_ipsi_cue_on_values)),
                            np.std(return_ipsi_cue_off_values) / np.sqrt(len(return_ipsi_cue_off_values))]


        #print('SOR_choice_peak_values: ' + str(SOR_choice_peak_values))
        #print('SOR_cue_peak_values: ' + str(SOR_cue_peak_values))

        for i in range(0, len(return_contra_cue_on_values)):
            x_val = [0, 1]
            y_val = [return_contra_cue_on_values[i], return_contra_cue_off_values[i]]
            ax3.plot(x_val, y_val, color='#0A3D62', linewidth=0.5, marker='o', markersize=10, label='Contra' if i == 0 else "")
            z_val = [return_ipsi_cue_on_values[i], return_ipsi_cue_off_values[i]]
            ax3.plot(x_val, z_val, color='#A9D6E5', linewidth=0.5, marker='o', markersize=10, label='Ipsi' if i == 0 else "")

            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.legend(loc='upper right', fontsize=8, frameon=False)


        #ax3.plot(x_val, contra_mean_peak_values, color='#062A45', linewidth=1, marker='o', markersize=10)
        #ax3.plot([0, 0], [contra_mean_peak_values[0] + contra_sem_peak_values[0], contra_mean_peak_values[0] - contra_sem_peak_values[0]],
        #         color='#062A45',
        #         linewidth=1)
        #ax3.plot([1, 1], [contra_mean_peak_values[1] + contra_sem_peak_values[1], contra_mean_peak_values[1] - contra_sem_peak_values[1]],
        #         color='#062A45',
        #         linewidth=1)

        #ax3.plot(x_val, ipsi_mean_peak_values, color='#5BA4C7', linewidth=1, marker='o', markersize=10)
        #ax3.plot([0, 0], [ipsi_mean_peak_values[0] + ipsi_sem_peak_values[0], ipsi_mean_peak_values[0] - ipsi_sem_peak_values[0]],
        #         color='#5BA4C7',
        #         linewidth=1)
        #ax3.plot([1, 1], [ipsi_mean_peak_values[1] + ipsi_sem_peak_values[1], ipsi_mean_peak_values[1] - ipsi_sem_peak_values[1]],
        #            color='#5BA4C7',
        #            linewidth=1)


        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["cue on", "cue off"])
        ax3.set_ylabel('Z-scored dF/F')
        ax3.set_xlim(-0.2, 1.2)
        ax3.set_ylim(-1, 2)
        ax3.yaxis.set_ticks([-1, 0, 1, 2])
        fig3.tight_layout(pad=2)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'SOR_' + add + 'DP_plot.pdf', dpi=300,
                    transparent=True)



        plt.show()








    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot == 'SOR_group_plot':
        print(plot)
        #mice = ['TS24', 'TS26', 'TS27', 'TS32', 'TS33', 'TS34']
        #dates = ['20230929', '20230918', '20231003', '20231113', '20231110', '20231031']
        # performance 2AC: 57% +- 4%

        #SFN plot 231111 & used in paper:
        add = 'classic_SOR_fig'
        mice = ['TS24', 'TS26', 'TS27','TS32','TS33','TS34']
        dates = ['20230929', '20230918','20231003', '20231026','20231102','20231031']
        # performance: 55% +- 4%

        #add = '1stExposureCue'
        #mice = ['TS32', 'TS33', 'TS34']
        #dates = ['20231026', '20231026', '20231026']


        x_range = [-2, 3]
        y_range = [-1, 2]

        fig1, ax1 = plt.subplots(3, len(mice)*2, figsize=(6 * len(mice), 10)) # each mouse

        all_group_data = {'SOR_cue':[], 'SOR_choice':[], 'SOR_reward':[], 'cue':[], 'choice':[], 'reward':[]}
        SOR_cue_peak_values = []
        SOR_choice_peak_values = []
        performances = []

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
            performances.append(data.performance)

            #print(vars(data))  # get all attributes of the data object

            alignements = ['SOR_choice', 'SOR_cue', 'SOR_reward', 'cue', 'choice', 'reward']

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
               #elif alignement == 'SOR_return_cueON':
                #    curr_data = data.SOR_return_cueON
                #elif alignement == 'SOR_return_cueOFF':
                 #   curr_data = data.SOR_return_cueOFF




                curr_data_mean = decimate(curr_data.contra_data.mean_trace, 10)
                curr_data_time = decimate(curr_data.contra_data.time_points, 10)
                curr_data_traces = decimate(curr_data.contra_data.sorted_traces, 10)

                curr_data_sem_upper = np.zeros(curr_data_mean.shape)
                curr_data_sem_lower = np.zeros(curr_data_mean.shape)
                curr_data_sem_lower, curr_data_sem_upper = calculate_error_bars(curr_data_mean,
                                                                                            curr_data_traces,
                                                                                            error_bar_method='sem')


                # get the peak values:
                if alignement == 'SOR_choice':
                    start_inx = 8000 # time 0
                    #SOR_choice_data_range = [start_inx:start_inx + 8000]
                    SOR_choice_time_range = curr_data_time[start_inx:start_inx + 8000]
                    SOR_choice_peak_index = np.argmax(curr_data_mean[start_inx:start_inx + 8000])
                    SOR_choice_peak_time = SOR_choice_time_range[SOR_choice_peak_index]
                    SOR_choice_peak_value = curr_data_mean[start_inx + SOR_choice_peak_index]
                    SOR_choice_peak_values.append(SOR_choice_peak_value)
                        #print(mouse + ' ' + date + ' ' + alignement + ' ' + str(SOR_choice_peak_value) + ' ' + str(SOR_choice_peak_time))

                elif alignement == 'SOR_cue':
                    SOR_cue_peak_value = curr_data_mean[start_inx + SOR_choice_peak_index]
                    SOR_cue_peak_values.append(SOR_cue_peak_value)


                # plot single mouse:
                legend = []
                if a < 3:   # SOR
                    c = 2*i
                    r=a
                    ax1[r, c].set_title(mouse + '_' + date, fontsize=10, loc='left')
                    legend.append(alignement)
                    color_mean = '#e377c2'
                    color_sem = '#e377c2'
                    alpha = 0.3
                elif a > 2 & a < 6:   # 2AC
                    c = 2*i + 1
                    r = a -3
                    legend.append('2AC_' + alignement)
                    color_mean = '#3F888F'
                    color_sem = '#7FB5B5'
                    alpha = 1
                elif a > 5: # SOR return movements with and without cue, contra and ipsi
                    c = 2*i + 1
                    r = 1
                    legend.append(alignement)
                    color_mean = '#e377c2'
                    color_sem = '#e377c2'
                    alpha = 0.3



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
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + '2AC_vs_SOR_group_RETURN.pdf', dpi=300,
                        transparent=True)


        print('performances: ' + str("%.2f" % np.mean(performances)) + ' +/- ' + str("%.2f" % (np.std(performances) / np.sqrt(nr_mice)))) # + ' n=' + str(nr_mice) + '(' + str(performances) + ')')




        fig2, ax2 = plt.subplots(3, 2, figsize=(6, 10))  # group average
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



        # dotplot
        fig3, ax3 = plt.subplots(1, 1, figsize=(2, 3))  # dotplot
        print('SOR_choice_peak_values: ' + str(SOR_choice_peak_values))
        print('SOR_cue_peak_values: ' + str(SOR_cue_peak_values))

        mean_peak_values = [np.mean(SOR_choice_peak_values), np.mean(SOR_cue_peak_values)]
        sem_peak_values = [np.std(SOR_choice_peak_values) / np.sqrt(len(SOR_choice_peak_values)),
                       np.std(SOR_cue_peak_values) / np.sqrt(len(SOR_cue_peak_values))]


        # Effect size based on glass delta method:
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444174/
        cohen_d_paired(SOR_choice_peak_values, SOR_cue_peak_values)

        # calculate p-value:

        # Shapiro-Wilk test for normality
        stat, p1 = shapiro(SOR_choice_peak_values)
        print('Shapiro-Wilk test for normality APE peak values: Statistics=%.3f, p=%.3f' % (
        stat, p1))  # p > 0.05, normal distribution
        stat, p2 = shapiro(SOR_cue_peak_values)
        print('Shapiro-Wilk test for normality RTC peak values: Statistics=%.3f, p=%.3f' % (
        stat, p2))  # p > 0.05, normal distribution

        if p1 > 0.05 and p2 > 0.05:
            # Paired t-test
            stat, p = ttest_rel(SOR_choice_peak_values, SOR_cue_peak_values)
            print('Paired t-test for normally distributed data: Statistics=%.3f, p=%.5f' % (
            stat, p))  # p < 0.05, reject null hypothesis, significant difference between groups
        else:
            # Kruskal-Wallis H-test
            stat, p = kruskal(SOR_choice_peak_values, SOR_cue_peak_values)
            print('Kruskal-Wallis H-test: Statistics=%.3f, p=%.3f' % (
            stat, p))  # p < 0.05, reject null hypothesis, significant difference between groups

        # Test if the mean of the data is significantly different from 0
        stat, p = ttest_1samp(SOR_cue_peak_values, popmean=0)
        print("SOR_cue_peak_value different from 0? t-statistic:", stat, "p-value:", p)

        stat, p = ttest_1samp(SOR_choice_peak_values, popmean=0)
        print("SOR_choice_peak_value different from 0? t-statistic:", stat, "p-value:", p)


        for i in range(0, len(SOR_choice_peak_values)):
            x_val = [0, 1]
            y_val = [SOR_choice_peak_values[i], SOR_cue_peak_values[i]]
            ax3.plot(x_val, y_val, color='#3F888F', linewidth=0.5, marker='o', markersize=10)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            # ax.set_xticks([0, 1], labels=["APE", "RTC"])

        ax3.plot(x_val, mean_peak_values, color='r', linewidth=1, marker='o', markersize=10)
        ax3.plot([0, 0], [mean_peak_values[0] + sem_peak_values[0], mean_peak_values[0] - sem_peak_values[0]], color='r',
                linewidth=1)
        ax3.plot([1, 1], [mean_peak_values[1] + sem_peak_values[1], mean_peak_values[1] - sem_peak_values[1]], color='r',
                linewidth=1)

        ax3.set_xticks([0, 1])
        ax3.set_ylabel('Z-scored dF/F')
        ax3.set_xlim(-0.2, 1.2)
        ax3.set_ylim(-1, 2)
        ax3.yaxis.set_ticks([-1, 0, 1, 2])
        fig3.tight_layout(pad=2)
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'SOR_choice_vs_cue_group_plot_all_mice_dotplot.pdf', dpi=300,
                transparent=True)
        plt.show()






   # ------------------------------------------------------------------------------------------------------------------
   # ------------------------------------------------------------------------------------------------------------------

    if plot == 'SOR_group_plot_NatRev':
        # Include incorrect trials, plot for NatRev
        print(plot)
        # mice = ['TS24', 'TS26', 'TS27', 'TS32', 'TS33', 'TS34']
        # dates = ['20230929', '20230918', '20231003', '20231113', '20231110', '20231031']
        # performance 2AC: 57% +- 4%

        # SFN plot 231111 & used in paper:
        mice = ['TS24', 'TS26', 'TS27', 'TS32', 'TS33', 'TS34']
        dates = ['20230929', '20230918', '20231003', '20231026', '20231102', '20231031']
        # performance: 55% +- 4%

        x_range = [-2, 3]
        y_range = [-1, 2]

        all_group_data = {'SOR_cue': [], 'SOR_choice': [], 'SOR_reward': [], 'cue': [], 'choice': [], 'reward': []}
        all_group_incorrect_data = {'corr_choice': [],'incorr_choice': [], 'xreward': []}


        SOR_cue_peak_values = []
        SOR_choice_peak_values = []
        performances = []

        for i, mouse in enumerate(mice):
            print('     >' + mouse + ' ' + dates[i])
            nr_mice = len(mice)
            date = dates[i]
            trial_data_path = main_directory + 'processed_data\\' + mouse + '\\'
            # get SOR data:
            experiment = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
            # print(experiment)
            fiber_side = experiment['fiber_side'].values[0]
            recording_site = experiment['recording_site'].values[0]
            data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)
            performances.append(data.performance)

            # print(vars(data))  # get all attributes of the data object

            alignements = ['SOR_choice', 'SOR_cue', 'SOR_reward', 'cue', 'choice', 'reward']

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

                all_group_data[alignement].append(curr_data_mean)

                # Adding incorrect reward aligned data
                if a == 1:

                    adding = ['xreward', 'incorr_choice', 'corr_choice']

                    for add in adding:
                        if add == 'xreward':
                            Xcurr_data = data.reward.contra_data_incorrect
                        elif add == 'incorr_choice':
                            Xcurr_data = data.choice.contra_data.incorrect
                            n_incorr = Xcurr_data.sorted_traces.shape[0]
                            # check where no data
                        elif add == 'corr_choice':
                            Xcurr_data = data.choice.contra_data.correct
                            n_corr = Xcurr_data.sorted_traces.shape[0]
                            trial_nums = data.choice.contra_data.correct.trial_nums
                            #print('corr trial nums: ' + str(trial_nums))


                        Xcurr_data_mean = decimate(Xcurr_data.mean_trace, 10)
                        Xcurr_data_time = decimate(Xcurr_data.time_points, 10)
                        Xcurr_data_traces = decimate(Xcurr_data.sorted_traces, 10)

                        Xcurr_data_sem_upper = np.zeros(Xcurr_data_mean.shape)
                        Xcurr_data_sem_lower = np.zeros(Xcurr_data_mean.shape)
                        Xcurr_data_sem_lower, Xcurr_data_sem_upper = calculate_error_bars(Xcurr_data_mean,
                                                                                        Xcurr_data_traces,
                                                                                        error_bar_method='sem')
                        all_group_incorrect_data[add].append(Xcurr_data_mean)

                    print('         > n_incorr = ' + ' ' + str(n_incorr) + ', n_corr = ' + str(
                        n_corr) + ', contra performance: ' + str("%.1f" % (n_corr / (n_incorr + n_corr) * 100) + '%'))


                # get the peak values:
                if alignement == 'SOR_choice':
                    start_inx = 8000  # time 0
                    # SOR_choice_data_range = [start_inx:start_inx + 8000]
                    SOR_choice_time_range = curr_data_time[start_inx:start_inx + 8000]
                    SOR_choice_peak_index = np.argmax(curr_data_mean[start_inx:start_inx + 8000])
                    SOR_choice_peak_time = SOR_choice_time_range[SOR_choice_peak_index]
                    SOR_choice_peak_value = curr_data_mean[start_inx + SOR_choice_peak_index]
                    SOR_choice_peak_values.append(SOR_choice_peak_value)
                    # print(mouse + ' ' + date + ' ' + alignement + ' ' + str(SOR_choice_peak_value) + ' ' + str(SOR_choice_peak_time))

                elif alignement == 'SOR_cue':
                    SOR_cue_peak_value = curr_data_mean[start_inx + SOR_choice_peak_index]
                    SOR_cue_peak_values.append(SOR_cue_peak_value)




        print('performances: ' + str("%.2f" % np.mean(performances)) + ' +/- ' + str("%.2f" % (
                    np.std(performances) / np.sqrt(
                nr_mice))))  # + ' n=' + str(nr_mice) + '(' + str(performances) + ')')

        fig2, ax2 = plt.subplots(3, 2, figsize=(6, 10))  # group average
        fig2.tight_layout(pad=4)
        a = 0
        align_to = ['cue', 'choice', 'reward']
        for a, key in enumerate(all_group_data.keys()):
            curr_data = all_group_data[key]
            # len(curr_data) = nr_mice
            # curr_data[0] = data points (after decimate! > 16000)

            curr_data_set_mean = np.mean(curr_data, axis=0)
            curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)
            # print(str(a) + ' ' + key) #' shape of data: ' + str(curr_data.shape))

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
                legend.append('2AC')  # ('2AC_' + key)
                color_mean = '#3F888F'
                color_sem = '#7FB5B5'
                alpha = 1

            ax2[0, 1].set_title('n = ' + str(nr_mice) + ' mice', fontsize=12, loc='right')
            ax2[r, c].plot(curr_data_time, curr_data_set_mean, lw=2, color=color_mean)
            ax2[r, c].fill_between(curr_data_time, curr_data_set_mean - curr_data_set_sem,
                                   curr_data_set_mean + curr_data_set_sem, color=color_sem,
                                   linewidth=1, alpha=alpha)

            # plot incorrect reward aligned data
            if key == 'reward':
                curr_data = all_group_incorrect_data['xreward']
                curr_data_set_mean = np.mean(curr_data, axis=0)
                curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)
                ax2[r, c].plot(curr_data_time, curr_data_set_mean, lw=2, color='red')
                ax2[r, c].fill_between(curr_data_time, curr_data_set_mean - curr_data_set_sem,
                                       curr_data_set_mean + curr_data_set_sem, color='red',
                                       linewidth=1, alpha=0.3)

            # plot in/correct choice aligned data
            if key == 'choice':
                curr_data = all_group_incorrect_data['corr_choice']
                curr_data_set_mean = np.mean(curr_data, axis=0)
                curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)
                ax2[r, c].plot(curr_data_time, curr_data_set_mean, lw=2, color='green')
                ax2[r, c].fill_between(curr_data_time, curr_data_set_mean - curr_data_set_sem,
                                       curr_data_set_mean + curr_data_set_sem, color='green',
                                       linewidth=1, alpha=0.3)

                curr_data = all_group_incorrect_data['incorr_choice']
                curr_data_set_mean = np.mean(curr_data, axis=0)
                curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)
                ax2[r, c].plot(curr_data_time, curr_data_set_mean, lw=2, color='red')
                ax2[r, c].fill_between(curr_data_time, curr_data_set_mean - curr_data_set_sem,
                                       curr_data_set_mean + curr_data_set_sem, color='red',
                                       linewidth=1, alpha=0.3)


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

        # plt.tight_layout()
        plt.savefig(main_directory + 'YJ_SummaryPlots\\' + '2AC_vs_SOR_group_plot_all_mice_NatRev.pdf', dpi=300,
                    transparent=True)
        # plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_vs_RTC_group_plot_all_mice.png', dpi=300, transparent=True)

        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


    if plot == 'APE_group_plot':
       DATVglut_mice = ['T10','T11','T12','T13']
       DATVglut_dates = ['20231106','20231107','20231103','20231102']

       dLight_mice = ['TS24','TS26','TS33','TS34']
       dLight_dates = ['20230920','20230918','20231102','20231030']



       mice = DATVglut_mice
       dates = DATVglut_dates
       group = 'DATVglut'

       mice = dLight_mice
       dates = dLight_dates
       group = 'dLight'


       x_range = [-2, 3]
       y_range = [-1, 2]

       all_group_data = {'cue_contra': [],'cue_ipsi': [], 'choice_contra': [],'choice_ipsi': [], 'reward_correct': [], 'reward_incorrect': []}
       fig1, ax1 = plt.subplots(3, 1, figsize=(4 , 10))

       for i, mouse in enumerate(mice):
            nr_mice = len(mice)
            date = dates[i]
            # get movement signal from same day session:
            experiment = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse)]
            fiber_side = experiment['fiber_side'].values[0]
            recording_site = experiment['recording_site'].values[0]
            data = get_SessionData(main_directory, mouse, date, fiber_side, recording_site)

            alignements = ['cue', 'choice', 'reward']

            for a, alignement in enumerate(alignements):
                if alignement == 'cue':
                    curr_data = data.cue
                elif alignement == 'choice':
                    curr_data = data.choice
                elif alignement == 'reward':
                    curr_data = data.reward

                if alignement is not 'reward':

                    # contalateral data:
                    curr_data_contra_mean = decimate(curr_data.contra_data.mean_trace, 10)
                    curr_data_contra_time = decimate(curr_data.contra_data.time_points, 10)
                    curr_data_contra_traces = decimate(curr_data.contra_data.sorted_traces, 10)

                    curr_data_contra_sem_upper = np.zeros(curr_data_contra_mean.shape)
                    curr_data_contra_sem_lower = np.zeros(curr_data_contra_mean.shape)
                    curr_data_contra_sem_lower, curr_data_contra_sem_upper = calculate_error_bars(curr_data_contra_mean,
                                                                                curr_data_contra_traces,
                                                                                error_bar_method='sem')

                    all_group_data[alignement + '_contra'].append(curr_data_contra_mean)

                    # ipsilateral data:
                    curr_data_ipsi_mean = decimate(curr_data.ipsi_data.mean_trace, 10)
                    curr_data_ipsi_time = decimate(curr_data.ipsi_data.time_points, 10)
                    curr_data_ipsi_traces = decimate(curr_data.ipsi_data.sorted_traces, 10)

                    curr_data_ipsi_sem_upper = np.zeros(curr_data_ipsi_mean.shape)
                    curr_data_ipsi_sem_lower = np.zeros(curr_data_ipsi_mean.shape)
                    curr_data_ipsi_sem_lower, curr_data_ipsi_sem_upper = calculate_error_bars(curr_data_ipsi_mean,
                                                                                              curr_data_ipsi_traces,
                                                                                              error_bar_method='sem')

                    all_group_data[alignement + '_ipsi'].append(curr_data_ipsi_mean)

                elif alignement is 'reward':
                    # correct trials:
                    correct_traces = np.concatenate((curr_data.contra_data.sorted_traces, curr_data.ipsi_data.sorted_traces), axis=0)

                    print(curr_data.contra_data.sorted_traces.shape[0])
                    print(curr_data.ipsi_data.sorted_traces.shape[0])
                    print(correct_traces.shape[0])
                    curr_data_correct_mean = decimate(np.mean(correct_traces, axis=0), 10)
                    curr_data_correct_time = decimate(curr_data.contra_data.time_points, 10) # time points are always the same, no matter ipsi / contra / in/correct
                    curr_data_correct_traces = decimate(correct_traces, 10)

                    curr_data_correct_sem_upper = np.zeros(curr_data_correct_mean.shape)
                    curr_data_correct_sem_lower = np.zeros(curr_data_correct_mean.shape)
                    curr_data_correct_sem_lower, curr_data_correct_sem_upper = calculate_error_bars(curr_data_correct_mean,
                                                                                                  curr_data_correct_traces,
                                                                                                  error_bar_method='sem')
                    all_group_data[alignement + '_correct'].append(curr_data_correct_mean)

                    # incorrect trials:
                    #incorrect_traces = curr_data.contra_data_incorrect.sorted_traces
                    incorrect_traces = np.concatenate((curr_data.contra_data_incorrect.sorted_traces, curr_data.ipsi_data_incorrect.sorted_traces), axis=0)
                    curr_data_incorrect_mean = decimate(np.mean(incorrect_traces, axis=0), 10)
                    curr_data_incorrect_time = decimate(curr_data.contra_data.time_points, 10) # time points are always the same, no matter ipsi / contra / in/correct
                    curr_data_incorrect_traces = decimate(incorrect_traces, 10)

                    #curr_data_correct_sem_upper = np.zeros(curr_data_correct_mean.shape)
                    #curr_data_correct_sem_lower = np.zeros(curr_data_correct_mean.shape)
                    #curr_data_correct_sem_lower, curr_data_correct_sem_upper = calculate_error_bars(curr_data_correct_mean,
                        #                                                                          curr_data_correct_traces,
                        #                                                                          error_bar_method='sem')
                    all_group_data[alignement + '_incorrect'].append(curr_data_incorrect_mean)

       #all_group_data = {'cue_contra': [], 'cue_ipsi': [], 'choice_contra': [], 'choice_ipsi': [], 'reward_correct': [],
        #                 'reward_incorrect': []}

       # plot data:
       fig1.tight_layout(pad=4)
       align_to = ['cue', 'choice', 'reward']
       legend = ['contra', 'ipsi']
       for a, key in enumerate(all_group_data.keys()):
                    color = 'cyan'
                    if a == 0:
                        r = 0
                        color = 'blue' # '#FF6495ED' # 'blue'
                    if a == 2:
                        r = 1
                        color = 'blue'
                    elif a == 4:
                        r = 2
                        color = 'blue'

                    curr_data = all_group_data[key]
                    curr_data_set_mean = np.mean(curr_data, axis=0)
                    curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)


                    ax1[r].plot(curr_data_contra_time, curr_data_set_mean, lw=1, color=color)
                    ax1[r].fill_between(curr_data_contra_time, curr_data_set_mean + curr_data_set_sem, curr_data_set_mean - curr_data_set_sem, color=color,
                                           linewidth=1, alpha=0.3)

       for r in range(3):
           if r < 2:
               legend = ['contra','ipsi']
               ax1[r].legend(legend, loc='upper right', fontsize=8, frameon=False)
           if r == 2:
                legend = ['correct', 'incorrect']
                ax1[r].legend(legend, loc='upper right', fontsize=8, frameon=False)

           ax1[r].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
           ax1[r].spines['top'].set_visible(False)
           ax1[r].spines['right'].set_visible(False)
           ax1[r].set_ylim(y_range)
           ax1[r].set_ylabel('Z-scored dF/F')
           #ax1[r].set_yticks(np.arange(y_range[0], y_range[1] + 1, 1))
           ax1[r].set_xlabel('Time (s)')
           ax1[r].set_xlim(x_range)
       plt.savefig(main_directory + 'YJ_SummaryPlots\\' + 'APE_group_plot_all_mice' + group + '.pdf', dpi=300, transparent=True)
       plt.show()















