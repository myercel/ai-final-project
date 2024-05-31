import numpy as np
import math
from sklearn.metrics import accuracy_score
from config import config
from hyperparameters import models
import os

# Return boolean arrays with length corresponding to n_samples
# the split is done based on the number of IDs
def split(ids, train, val, test):
    assert (train+val+test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.isin(ids, IDs[:train_split])
    val = np.isin(ids, IDs[train_split:train_split+val_split])
    test = np.isin(ids, IDs[train_split+val_split:])

    return train, val, test

def train_cnn(trainX, trainY, ids, shuffle_time=False, shuffle_electrodes=False):

    if shuffle_time:
        assert config['feature_extraction'] == False, "Shuffling is not allowed on extracted featrues"
        assert shuffle_electrodes == False, "Shuffling is only allowed on one dimension (time or electrodes)"
       
        # Shuffle accross time
       
        print("Shuffling accross time axis")
        trainX = np.transpose(trainX, (0, 2, 1))

        permutation = np.random.permutation(trainX.shape[2])
        for i in range(trainX.shape[0]):
            for j in range(trainX.shape[1]):
                trainX[i, j, :] = trainX[i, j, permutation]

        # Transpose back to the original shape
        trainX = np.transpose(trainX, (0, 2, 1))
    
    if shuffle_electrodes:
        assert config['feature_extraction'] == False, "Shuffling is not allowed on extracted featrues"
        assert shuffle_time == False, "Shuffling is only allowed on one dimension (time or electrodes)"
        
        print("Shuffling accross electrodes axis")
        
        permutation = np.random.permutation(trainX.shape[2])
        for i in range(trainX.shape[0]):
            for j in range(trainX.shape[1]):
                trainX[i, j, :] = trainX[i, j, permutation]

    # Split data into training, validation and testing
    train, val, test = split(ids, 0.7, 0.15, 0.15)
    X_train, y_train = trainX[train], trainY[train]
    X_val, y_val = trainX[val], trainY[val]
    X_test, y_test = trainX[test], trainY[test]

    # dummy value before running anything
    score = -1

    # In the adapted version of the code, models will only include the CNN
    for name, model in models.items():
        # create the model with the corresponding parameters
        trainer = model[0](**model[1])

        # Saving Trainer into a path is necessary for the code to exit gracefully
        # So, we are creating a dummy path to save into
        path = './dummyPath/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Fit the model on the X_train and y_train
        trainer.fit(X_train, y_train, X_val, y_val)

        # Save trainer ifnormation
        trainer.save(path)

        # Define scorign function
        scoring = (lambda y, y_pred: accuracy_score(y, y_pred.ravel()))

        # Get final score on the testing set
        score = scoring(y_test, trainer.predict(X_test))  

    return score

def benchmark(trainX, trainY):

    print("Shape of trainX before slicing: ", trainX.shape)
    print("Shape of trainY before slicing: ", trainY.shape)

    # Only the first 10k data points due to computational limitation
    trainX = trainX[:10000]
    trainY = trainY[:10000]

    # Printing to validate that we are using the reduced size dataset

    print("Shape of trainX after slicing: ", trainX.shape)
    print("Shape of trainY after slicing: ", trainY.shape)

    
    if config['task'] == 'LR_task' and config['dataset'] == 'antisaccade':
        ids = trainY[:, 0] # The first column are the IDs
        y = trainY[:,1] # The second column are the labels, we take the second

        score = train_cnn(trainX=trainX, trainY=y, ids=ids, 
        shuffle_electrodes = config['shuffle_electrodes'], shuffle_time = config['shuffle_time'])

        print("Final Model's score on testing set", score)
    else:
        raise NotImplementedError(f"This benchmark only works for LR_task on antisaccade data")