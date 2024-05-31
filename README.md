## Project Overview

In this project, we are interested in examining whether time-series data in the form of EEG brainwave recordings have local spatial relations, similar to those found in imgaes, that would allow a Convolutional Neural Network to perform well on an EEG classification task.

Our data set and code comes from the [EEGEyeNet paper](https://arxiv.org/pdf/2111.05100v2) and involves Left/Right antisaccade classification, in which participants were instructed to look the opposite way from the direction they were instructed with. While this paper examined the performance of a variety of models, we are specifically interested in the CNN. The CNN model from this paper is located in `EEGEyeNet/DL_Models`. 

NOTE: We had to create a `dummyPath` folder in order to have a location to save the trained CNN. This is necessary for the code to exit succesfully.

## Dataset Locations

The dataset we used for this project is the original time-series EEG dataset that is quite large (15GB). Due to the size of this dataset, it is stored locally on the CS lab computer **Dratini** at `/local/ayahia1`.

If you would like to download this dataset to a different directory, change `config['data_dir']` in `config.py` to the
new data directory location.

## Environment

Our code needs to be run in an environment that can support tensorflow. On the CS lab machines, run `source /usr/swat/bin/CS63env` before running our code.

## Running Code

`benchmark.py` contains the code for splitting the dataset into a training, validating, and testing set, as well as training and testing the model. Within this file, we had to limit the size of our dataset to the first 10,000 EEG recordings as tensorflow was not able to handle a larger amount of data. 

The following changed can be made before running the code:

1. Within `config.py`:
  - `shuffle_time` can be set to **True** or **False**. If set to true, the time-series data within each EEG recording will be shuffled.
  - `shuffled_electrodes` can be set to **True** or **False**. If set to True, the association between an electrode and an EEG recoding will be shuffled. This equates to shuffling the locations of the electrodes on the EEG net on a participant's head.

    **NOTE: The data can only be shuffled on one dimesion at a time, cannot shuffle on time and electrodes simultaneously.**
  
 2. Within `hyperparameters.py`:
  - `batch_size` can be changed to a larger/smaller value. This is the number of EEG recordings trained on during each epoch, as we do not train on the entire dataset in each epoch.
  - `depth` can be changed to allow for a shallower/deeper network. We changed the depth to 4 from the original depth of 12 in the EEGEyeNet paper to avoid overfitting.
  - `epochs` can be changed to allow the model to train for longer/shorter.

To run our code with the above specified hyperparameters, run `python3 main.py`.
