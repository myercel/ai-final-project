# configuration used by the training and evaluation methods
config = dict()

##################################################################
##################################################################
############### CONFIGURATIONS #########################
##################################################################
##################################################################
config['task'] = 'LR_task'
config['dataset'] = 'antisaccade'
config['preprocessing'] = 'max'


config['feature_extraction'] = False # 1. If set to False, models operate on the time series data
                                     # 2. If set to True, models operate on features extracted from time
                                     # series data
config['shuffle_time'] = False 
config['shuffle_electrodes'] = False

##################################################################
##################################################################
############### PATH CONFIGURATIONS ##############################
##################################################################
##################################################################

# Path to folders with the data
config['data_dir'] = '/local/ayahia1/'

def build_file_name():
    all_EEG_file = config['task'] + '_with_' + config['dataset']
    all_EEG_file = all_EEG_file + '_' + 'synchronised_' + config['preprocessing']
    all_EEG_file = all_EEG_file + ('_hilbert.npz' if config['feature_extraction'] else '.npz')
    return all_EEG_file
    
config['all_EEG_file'] = build_file_name() 


##################################################################
##################################################################
############### MODELS CONFIGURATIONS ############################
##################################################################
##################################################################
# Specific to models now
config['learning_rate'] = 1e-4

##################################################################
##################################################################
############### HELPER VARIABLES #################################
##################################################################
##################################################################
config['trainX_variable'] = 'EEG'
config['trainY_variable'] = 'labels'