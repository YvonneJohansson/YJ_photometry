
from YJ_preprocessing_basic_functions import *


if __name__ == "__main__":
    folder_path = 'Z:\\users\\Yvonne\\photometry_2AC\\'  # CEPH folder with raw data
    #folder_path = 'Z:\\users\\francesca\\photometry_2AC\\'
    saving_folder_path = 'Z:\\users\\Yvonne\\photometry_2AC\\'

    #mouse_ids = ['SNL_photo21','SNL_photo22','SNL_photo23','SNL_photo24','SNL_photo25','SNL_photo26']
    mouse_ids = ['T16'] #TS21 TS18_30

    dates = ['20231130']


    #force_protocol = 'Random_Tone_Clouds'
    #force_protocol = 'Random_WN'
    force_protocol = 'Airpuff'
    #force_protocol = ''


    for date in dates:
        for mouse_id in mouse_ids:
            all_experiments = get_all_experimental_records()
            experiments_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
            csv_protocol = experiments_to_process['experiment_notes'].values[0]
            #print(csv_protocol)
            if force_protocol is 'Random_Tone_Clouds':
                protocol = force_protocol
            elif force_protocol is 'Random_WN':
                protocol = 'Random_WN'
            elif force_protocol is 'Airpuff':
                protocol = 'AirpuffPhoto'
            elif csv_protocol == 'SOR':
                protocol = 'Two_Alternative_Choice_Tones_On_Return'
            else:
                protocol = 'Two_Alternative_Choice'
            experiments_to_process['date'] = date
            print(">> Now processing: " + date + '_' + mouse_id + '_' + protocol)
            #pre_process_experiments(experiments_to_process, method='pyphotometry', protocol='Two_Alternative_Choice')
            pre_process_experiments(experiments_to_process, method='lerner', protocol= protocol, folder_path=folder_path, saving_folder_path=saving_folder_path)
