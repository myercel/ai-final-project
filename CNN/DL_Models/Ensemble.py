"""
Common interface to fit and predict with torch and tf ensembles 
"""
from DL_Models.tf_models.Ensemble_tf import Ensemble_tf
import numpy as np

class Ensemble:

    def __init__(self, model_name, nb_models, loss, batch_size=64, **model_params):
        """

        :type model_params: dic
        """
        self.type = 'classifier'
        self.ensemble = Ensemble_tf(model_name=model_name, nb_models=nb_models, loss=loss, batch_size=batch_size, **model_params)
    
    def fit(self, trainX, trainY, validX, validY):
        self.ensemble.fit(trainX, trainY, validX, validY)
        
    def predict(self, testX):
        if self.type == 'classifier':
            return np.round(self.ensemble.predict(testX))
        else:
            return self.ensemble.predict(testX)

    def save(self, path):
        self.ensemble.save(path)

    def load(self, path):
        self.ensemble.load(path)