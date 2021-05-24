#!/usr/bin/env python

"""
===========================================================
Assignment 4: Neural Network Classification Benchmarks
===========================================================

Neural networks have allowed us to dive into the intricacies of deep learning and can provide one of the most accurate forms of classification. The model built in this script is a mere tip-toe into this field, as it is very simple (but effective!). 

The model uses the digits dataset from sklearn to classify digits into their correct numerical group (0:9) 
The output of the script is the evaluation metrics which are printed to the terminal.

This script will use argparse arguments to enable it to be run and the parameters ammended from the commandline.
It employs bonus challenge 1 (saving the output reports to an output file) and 2) determining number of hidden layers.
Instructions on how to use these can be found in the README.md file attached to this assignment. 

"""

"""
=======================
Import the Dependencies
=======================

"""
#operating systems
import os
import sys
sys.path.append(os.path.join(".."))

#Image manipulation 
import numpy as np

#command line functionality
import argparse

# Neural Network utility functions  
from Classification_benchmarks.utils.neuralnetwork import NeuralNetwork

#Neural network dependencies and data from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import metrics


"""
=============
Main Function
=============

"""
def main():
    
    print ("Hey there, let's get your Neural Network Classifier all set up.\n") 
    
    """
    =================================
    Setting up the argparse arguments
    =================================
    """
    # initialise argument parser 
    ap = argparse.ArgumentParser()
    
    # Command line argument interface (There are 2 mandatory arguments and 3 options) 
    ap.add_argument("-t", "--test_size", 
                    required = True, 
                    help = "decimal between 0 - 1 indicating what the test split should be")
    ap.add_argument("-e", "--epochs", 
                    required = False, 
                    default = 10, 
                    type = int,
                    help = "Specify the number of epochs to run with. The default is set to 10")
    ap.add_argument("-l", "--layers", 
                    required = False, 
                    default = 3, 
                    type = int,
                    help = "Specify number of hidden layers. Maximum recommended = 3, default = 2")
    ap.add_argument("-f", "--filename", 
                    required = False, 
                    default = "neural_network_report.csv",
                    help = "str indicating the filename of the output classification report")
    ap.add_argument("-o", "--output_path", 
                    required = True, 
                    help = "Path to output directory")
 

    # parse arguments (to make arguments usable within the script)   
    args = vars(ap.parse_args())
    
    
    """
    Assigning our arguments to variable names for the script
    """
   
    test_size = args["test_size"]
    epochs = args ["epochs"]
    layers = args["layers"]
    filename = args ["filename"]
    
    #output_path
    output_path = args["output_path"]
    
    #New users may want to create a new output directory, which we'll create here using the name defined in the terminal 
    #The code reads, "if an output path doesn't exist, please create one using os.mkdir()"
    if not os.path.exists(output_path):   
        os.mkdir(output_path) 
        
    
    """
    ============================== 
    Fetching the MNIST from opemml
    ==============================
    """
    
    print("I'm just about to fetch the data...") 
    
    # Call the data from open_ml 
    # pixels = pixel intensity values, labels = digit category they belong to
    pixels, labels = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    """
    ======================= 
    Pre-processing the data
    =======================
    """
    
    print("The data's looking good! Let's jump into the preprocessing...") 
    
    #Safety check: ensure the data is in a numpy array format
    pixels = np.array(pixels)
    labels = np.array(labels)

    # Conduct MinMax regularization on the pixel intensities (this ensures all numbers are between 0 - 1) 
    pixels_scaled = (pixels - pixels.min())/(pixels.max() - pixels.min())
    
    #Create train and test sets based on the args "split" defined by the user (using sklearn train_test_split function)
    X_train, X_test, y_train, y_test = train_test_split(pixels_scaled,
                                                        labels, 
                                                        random_state= 42, 
                                                        test_size= test_size)
    
    # Binarize the labels  
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    
    #Let the user know the training is about to begin 
    print("\nPreprocessing complete. I'm about to start training the network...")
     
    """
    ===========================
    Hidden layers functionality
    =========================== 
    """
    # For this we can use an if else statement for up to 3 layers (beyond that doesn't make sense)
    
    # Here, we'll define the steps for each layer  
    if (layers==1):
        nn = NeuralNetwork([X_train.shape[1], 10])
    elif (layers==2):
        nn = NeuralNetwork([X_train.shape[1], 16, 10])
    elif (layers==3):
        nn = NeuralNetwork([X_train.shape[1], 32, 16, 10])
    else:
        nn = NeuralNetwork([X_train.shape[1], 16, 10])
        
   
    """
    ==============================
    Fitting and training the model
    ==============================
    """   
    
    print("Still training... this might take a little while!\n") 
    
    #fit the model 
    print("[INFO] {}".format(nn))
    nn.fit(X_train, y_train, epochs=epochs)
    
    
    """
    ====================
    Evaluating the model
    ====================
    """ 
    # Update the user on progress 
    print(["Training complete, nice one. Let's take a look at the results...\n\n"])
    
    #Create the predictions 
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    
    #Print the performance metrics to the screen
    nn_report = (classification_report(y_test.argmax(axis=1), predictions))
    print(nn_report) 
    
    """
    ================================
    Saving the classification report
    ================================
    """ 
    
    #Then save a copy of this report to the Output_path 
    output_path = os.path.join(output_path, filename)
    # save metrics to path
    with open(output_path, "w") as report:
        report.write(f"Neural Network Classification Report:\n\n{nn_report}")
    
        
    #Then let the user know the script is complete and where the results can be found
    print(f"That's you all done - woohoo! The classification report has been saved in {output_path}.\n ") 
    

if __name__=="__main__":
    #execute main function
    main()
