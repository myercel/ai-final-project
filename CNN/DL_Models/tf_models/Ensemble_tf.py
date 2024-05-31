import tensorflow as tf
from config import config
from utils.utils import *
import keras
import logging
import os
import re 
import numpy as np 
from DL_Models.tf_models.CNN.CNN import CNN

class Ensemble_tf:
    """
    The Ensemble is a model itself, which contains a number of models that are averaged on prediction. 
    Default: nb_models of the model_type model
    Optional: Initialize with a model list, create a more versatile Ensemble model
    """
    def __init__(self, model_name='CNN', nb_models='5', loss='mse', batch_size=64, **model_params):
        """
        model_name: the model that the ensemble uses
        nb_models: Number of models to run in the ensemble
        model_list: optional, give a list of models that should be contained in the Ensemble
        ...
        """
        self.model_name = model_name
        self.nb_models = nb_models
        self.model_params = model_params
        self.batch_size = batch_size
        self.loss = loss
        self.model_instance = None
        self.load_file_pattern = re.compile(self.model_name[:3] +  '.*_nb_._best_model.pth', re.IGNORECASE)
        self.models = []
        self.model = CNN


    def fit(self, trainX, trainY, validX, validY):
        """
        Fit all the models in the ensemble and save them to the run directory 
        """    
        self.models = []
        # Fit the models 
        for i in range(self.nb_models):
            model = self.model(loss=self.loss, model_number=i, batch_size=self.batch_size, **self.model_params)
            self.models.append(model )
            model.fit(trainX, trainY, validX, validY)

    def predict(self, testX):
        pred = None
        for model in self.models:
            if pred is not None:
                pred += model.predict(testX)
            else:
                pred = model.predict(testX)
        return pred / len(self.models)

    def save(self, path):
        for i, model in enumerate(self.models):
            ckpt_dir = path + self.model_name + '_nb_{}_'.format(i)
            model.save(ckpt_dir)

    def load(self, path):
        self.models = []
        for file in os.listdir(path):
            if not self.load_file_pattern.match(file):
                continue
            self.models.append(keras.models.load_model(path+file))