import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt
import os


# Data class from data_gen.py
# Handles reading data from datafile 


class Data():
    
    def __init__(self, datafile, num_train, num_test):
        # Get directory 
        directory = os.getcwd()
        data_directory = os.path.join(directory, datafile)
        # Read data using pandas and storing as float 32
        data = pd.read_csv (r'{0}'.format(data_directory), delimiter = ' ', header = None, dtype = np.float32)   
        # Convert to numpy array
        data_array = np.array(data)
        # Randomly split batch into a test batch and train batch 
        train, test = train_test_split(data_array, test_size = num_test, train_size = num_train)
        
        train_image = train[:,:-1]/255
        test_image = test[:,:-1]/255
        
        # Store images and labels
        # Images are stored as a (num, 1, pixels, pixels) array
        self.y_train = train[:,-1] / 2
        self.x_train = np.reshape(train_image,(train_image.shape[0],1,int(sqrt(train_image.shape[1])),int(sqrt(train_image.shape[1]))))
        self.y_test = test[:,-1] / 2
        self.x_test= np.reshape(test_image,(test_image.shape[0],1,int(sqrt(test_image.shape[1])),int(sqrt(test_image.shape[1]))))
        
