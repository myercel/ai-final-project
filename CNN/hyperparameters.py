from DL_Models import Ensemble
from DL_Models.Ensemble import Ensemble
from config import config


# Values that can be changed to adjust the CNN behavior
batch_size = 64
depth = 4   # I changed this from 12 to 4 to avoid overfitting on the size-reduced dataset
epochs = 30 # I changed this to a small number for testing purposes


input_shape = (1, 258) if config['feature_extraction'] else (500, 129)
verbose = True

models = dict()
models['CNN'] = [Ensemble, {'model_name': 'CNN', 'nb_models' : 1, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}]