#!/usr/bin/env python

"""
===========================================================
Assignment 4: Logistic Regression Classification Benchmarks
===========================================================

Classification is a key feature of visual analytics and continues to develop year on year. It has been used in many fields from medical imaging (to help determine diagnosis), to business, to internet security, and beyond! 

This first script uses multinomial logistic regression to classify images of handwritten digits into the number category they belong to (between 0:9). The output of the script is the evaluation metrics which are printed to the terminal.

The script will use argparse arguments to enable it to be run and the parameters ammended from the commandline. Instructions of how to use these can be found in the README.md file attached to this assignment. 

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

#command line functionality
import argparse

# image manipulation and utility functions
import pandas as pd
import numpy as np
import Classification_benchmarks.utils.classifier_utils as clf_util

#Sklearn metrics  
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""
=====================================
Main Function with Argparse arguments
=====================================

"""
def main():
    """
    Setting up our arguments with argparse
    """
    # Initialise argument parser 
    ap = argparse.ArgumentParser()
    
    # Create the command line arguments  
    ap.add_argument("-t", "--test_size", 
                    required = True, 
                    type = float,
                    help = "decimal between 0 and 1 indicating what the test split should be")
    ap.add_argument("-o", "--output_path", 
                    required = True, 
                    help = "Path to output directory")
    ap.add_argument("-f", "--filename", 
                    required = False, 
                    default = "logistic_regression_classification_report.csv",
                    help = "str indicating the filename of the output classification report") 
    
    # parse arguments  
    args = vars(ap.parse_args())
    
    
    """
    Assigning our arguments to variable names for the script
    """
    #split  
    test_size = args["test_size"]
    
    #output_path
    output_path = args["output_path"]
    
    #New users may want to create a new output directory, which we'll create here using the name defined in the terminal 
    #The code reads, "if an output path doesn't exist, please create one using os.mkdir()"
    if not os.path.exists(output_path):   
        os.mkdir(output_path) 
        
    #Optional filename 
    filename = args["filename"]
        
         
    """
    ===================================    
    Fetching the MNIST data from opemml
    ===================================
    """
    
    print("Hey there, I'm just fetching the data...")
    
    # Call the data from open_ml 
    # pixels = pixel intensity values, labels = digit category they belong to  
    pixels, labels = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)    

    
    """
    ======================
    Preprocessing the data
    ======================
    """
    
    print("Looks good, now for the preprocessing...")
    
    #Safety check: ensure the data is in a numpy array format
    pixels = np.array(pixels)
    labels = np.array(labels)
    
    #Create train and test sets based on the args "split" defined by the user (using sklearn train_test_split function) 
    X_train, X_test, y_train, y_test = train_test_split(pixels, 
                                                        labels, 
                                                        random_state=33,
                                                        test_size= test_size ) #split will be a decimal between 0 - 1 
    
    #Scale the features to be between the scale of 0 to 1
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0

    """
    ===================================
    Train the Logistic Regression Model 
    ===================================
    """    
        
    """
    This model uses the SAGA algorithm which works well when the no. of samples is significantly larger than the no. of features
    It is able to finely optimize non-smooth objective functions which is useful when we use the l1-penalty.
    """
    
    print("The data's ready to go! \n\nNow let's build your model, this won't take too long.") 
    
    #Here, we use the scikit-learn LogisticRegression function to generate the model
    model = LogisticRegression(penalty='none', 
                         tol=0.1, 
                         solver='saga',           
                         multi_class='multinomial').fit(X_train_scaled, y_train)
    
    
    """
    ========================
    Calculate the accuracy
    ========================
    """
    
    # Predict the accuracy of the test dataset using the trained model 
    predictions = model.predict(X_test_scaled)

    """
    ================================
    Create the classification report 
    ================================
    """
    # Use sklearn to make a classification report & print to the terminal 
    cm = metrics.classification_report(y_test, predictions)
    print("Classification complete, nice! \nHere are your results...\n\n") 
    print(cm)
    
    #Then save a copy of this report to the Output_path 
    output_path = os.path.join(output_path, filename)
    # save metrics to path
    with open(output_path, "w") as report:
        report.write(f"Logistic Regression Classification Report:\n\n{cm}")

    #Finally, let the user know the script has been completed 
    print(f"That's you complete - woohoo! The classification report has been saved in {output_path}.\n ") 
    

#Close your main function 
if __name__=="__main__":
    #execute main function
    main()
