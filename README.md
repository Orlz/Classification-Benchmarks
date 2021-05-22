
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  [![describtor - e.g. python version](https://img.shields.io/badge/Python%20Version->=3.6-blue)](www.desired_reference.com)   ![](https://img.shields.io/badge/Software%20Mac->=10.14-pink)

# Classification Benchmarks

**Comparing a Logistic Regression and simple Neural Network Classifier on MNIST data**



<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/neural.png" width="100" height="100"/></div>




__Background to the assignment__

This assignment introduces us to a new method called classification. It takes our focus away from the simple manipulation of images and finding features within these images and moves us closer to a machine learning approach. Here, we look at how to build classifiers which are able to use features to predict the category membership of an image within a dataset. More specifically, we compare between classifiers to see which style of classification drives the most accurate results, a neural network or a logistic regression model. 


Classification is one of the most commonly used machine-learning techniques and grows in influence every year. It is a form of supervised learning whereby algorithms are used to look for patterns in the data, which can be used to predict the images category label. The model works by taking a training set (usually between 70 – 80% of the dataset) and learning which features and qualities in these images are important for assigning the image to its category label. The remaining data, known as the test set, is then run through the model. If the model is successful at learning which features are important to classify the image category with the training set, then it should be able to accurately predict which category this unseen image belongs to.

## Table of Contents 

- [Scripts and Data](#Scripts)
- [Methods](#Methods)
- [Operating the Scripts](#Operating)
- [Discussion of results](#Discussion)




## Scripts and Data


The assignment takes the form of 2 python scripts which should be called dirctly from the commandline. 


**lr-mnist.py**  This script uses multinomial logistic regression to classify images of handwritten digits into their correct digit group (0:9)



**nn-mnist.py**  This script employs a neural network to classify the same images of handwritten digits into their correct digit group (0:9)


Each script has a number of ammendable parameters which can be set in the command line. These are described below. 



__Data__ 
The data used is the MNIST data found on openml. It is pulled directly from open online sources and so no additional data needs to be added into the script.


This is a subset of the larger NIST file which contains 60,000 images of handwritten digits with 784 features. 
Information regarding this dataset can be found [here](https://www.openml.org/d/554)




## Methods


This problem was interested in creating simple classification benchmarks from which model evaluation could be considered. To do this, two classifier scripts were built to classify images of handwritten digits: a logistic regression classifier and a simple neural network classifier. In both scripts, the data was pulled directly from sklearn using the fetch_openml() function, transformed into a numerical NumPy array, and split into a train and test set using  sklearns’s train_test_split function. The pixels were then scaled to lie between 0 and 1, to enable convergence within the model. 


The multinomial logistic regression model was built using sklearn’s LogisticRegression() function with the saga algorithm used, which works well when the number of samples is significantly larger than the number of features - as is the case here. Predictions were made using sklearn’s predict functions and the classification_report() function created a readable report of the classifiers accuracy, which was stored in the out folder. 


The neural network classifier had an additional step of pre-processing where the labels were binarized and argparse arguments created the ability for the user to choose how many hidden layers the model was to be run with. The model was then fit using the NeuralNetwork class found in the neural_network.py script (found in the utils folder), and predictions were made using the predict function found in the same script. A report for this model was also created using sklearn’s classification_report function. 



## Operating the Scripts


There are 3 steps to take to get your script up and running:
1. Clone the repository 
2. Create a virtual environment (Computer_Vision02) 
3. Run the 2 scripts using command line parameters


___Output will be saved in a new folder called Output___


#### 1. Clone the repository

The easiest way to access the files is to clone the repository from the command line using the following steps 

```bash
#clone repository as classification_benchmarks_orlz
git clone https://github.com/Orlz/Classification-Benchmarks.git

```


#### 2. Create the virtual environment

You'll need to create a virtual environment which will allow you to run the script using all the relevant dependencies. This will require the requirements.txt file attached to this repository. 


To create the virtual environment you'll need to open your terminal and type the following code: 


```bash
bash create_virtual_environment.sh
```
And then activate the environment by typing: 
```bash
$ source Computer_Vision02/bin/activate
```


#### 3. Run the Scripts


There are 2 scripts to run. Each has a number of command-line parameters which can be set by the user. The options for these are as follows: 




### Logistic Regression Classifier: Script lr-mnist.py


___Parameter options = 3___


```
Letter call   | Are             | Required?| Input Type   | Description
------------- |:-------------:  |:--------:|:-------------:
`-t`          | `--test_size`   | Yes      | Float        | Decimal between 0 and 1 indicating the test size        |
`-o`          | `--output_path` | Yes      | String       | String indicating path to the output directory          |
`-f`          | `--filename`    | No       | String       | String indicating what the output file should be called |
```


___default filename: logistic_regression_classification_report.csv___




### Neural Network Classifier: Script nn-mnist.py


___Parameter options = 5___

```
| Letter call   | Are             | Required?| Input Type   | Description
| ------------- |:-------------:  |:--------:|:-------------:
|`-t`           | `--test_size`   | Yes      | Float        | Decimal between 0 and 1 indicating the test size              |
|`-o`           | `--output_path` | Yes      | String       | String indicating path to the output directory                |
|`-f`           | `--filename`    | No       | String       | String indicating what the output file should be called       |
|`-e`           | `--epochs`      | No       | Integer      | Integer indicating number of epochs to run with (default: 10) |
|`-l`           | `--layers`      | No       | Integer      | Integer indicating the number of hidden layers (default: 3)   |
```


Below is an example of the command line arguments for the logistic regression model with an 80:20 split: 


```bash
python3 src/lr-mnist.py -t 0.2 
```


## Discussion of Results 


The two models performed well on the MNIST dataset for classifying the images into their correct digit category. The neural network had a weighted average of 94%, which just marginally outperformed the logistic regression model which had a weighted average of 92%. The logistic regression model varied more in its classification between the classes, struggling to classify the digits 8 and 5 the most. The neural network also varied but was able to classify all digits with an F1 score in the 90 – 100% margin. It was particularly good at classifying 0’s and 1s, however this was also true for the logistic regression model which had equal or better scores for these digits. 


The high performance of these models is unsurprising considering the standardised way the images are presented, with little background noise and the digits being the key element of the image. It would be much more difficult for the classifier to separate these images if the digit was only part of a more complex picture with colours and other objects. Nevertheless, the models demonstrate the ability of computer vision to quickly learn complex contour patterns and extract the defining features of each to make highly accurate classification predictions. 




