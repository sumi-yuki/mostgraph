# -*- coding: utf-8 -*-
"""
Created by Yuki Sumi on Jan  23 2023

"""
# installing modules

# numpy https://numpy.org
# To install,
# pip install numpy
#  or
# conda install numpy

# pandas https://pandas.pydata.org
# To install,
# pip install pandas
#  or
# conda install -c conda-forge pandas

# Matplotlib https://matplotlib.org
# To install,
# pip install matplotlib
#  or
# conda install -c conda-forge matplotlib

# scikit-learn https://scikit-learn.org/stable/
# To install,
# pip install -U scikit-learn 
#  or
# conda install -c conda-forge scikit-learn

# imbalanced-learn https://imbalanced-learn.org/stable/index.html
# To install,
# pip install -U imbalanced-learn
#  or
# conda install -c conda-forge imbalanced-learn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization #, Dropout
from tensorflow.keras.models import Sequential

# the number of test data for the metrics
number_of_test = 20

# input FOT measured file name for normal subjects
MeasuredValueFileNameNormal = "control.csv"
# input FOT measured file name for asthmatic subjects
MeasuredValueFileNameAsthmatics = "ba.csv"
# List of measured Item used to analyse
MeasuredItemList = ["diagnosis", "gender", "R5in", "R5ex", "R20in", "R20ex", "X5in", "X5ex", "Fresin", "Fresex", "ALXin", "ALXex"]
# MeasuredItemList = ["diagnosis", "gender", "R5", "R5in", "R5ex", "R20", "R20in", "R20ex", "R5-R20", "R5-R20in", "R5-R20ex", "X5", "X5in", "X5ex", "Fres", "Fresin", "Fresex", "ALX", "ALXin", "ALXex"]
input_length = len(MeasuredItemList) - 1 # training data input are without label

# read the Mostgraph measurement results of respiratory normal subjects
# Column of csv file used are "diagnosis", "gender", "R5in", "R5ex", "R20in", "R20ex", "X5in", "X5ex", "Fresin", "Fresex", "ALXin", "ALXex"
mostgraphdata = pd.read_csv(MeasuredValueFileNameNormal, usecols = MeasuredItemList, encoding="utf-8")
# in case of selecting by sex
# mostgraphdata = mostgraphdata[mostgraphdata["gender"] == 1]
# mostgraphdata = mostgraphdata.drop("gender", axis=1)
# input_length = input_length - 1  # remove gender

# pandas -> numpy
mostgraph_total_data_numpy = mostgraphdata.values
# the number of data read
number_of_normal_mostgraphdata = mostgraph_total_data_numpy.shape[0]
# shuffle to make sequence random
np.random.shuffle(mostgraph_total_data_numpy)
# extract test data
mostgraph_normal_test_data_numpy = mostgraph_total_data_numpy[0:number_of_test]
# extract train and validation data
mostgraph_normal_train_data_numpy = mostgraph_total_data_numpy[number_of_test:number_of_normal_mostgraphdata]

# read the Mostgraph measurement results of asthmatics subjects
# Column of csv file used are "diagnosis", "gender", "R5in", "R5ex", "R20in", "R20ex", "X5in", "X5ex", "Fresin", "Fresex", "ALXin", "ALXex"
mostgraphdata = pd.read_csv(MeasuredValueFileNameAsthmatics, usecols = MeasuredItemList, encoding="utf-8")
# in case of selecting by sex
# mostgraphdata = mostgraphdata[mostgraphdata["gender"] == 1]
# mostgraphdata = mostgraphdata.drop("gender", axis=1)

# pandas -> numpy
mostgraph_total_data_numpy = mostgraphdata.values
# shuffle to make sequence random
np.random.shuffle(mostgraph_total_data_numpy)
# the number of data read
number_of_abnormal_mostgraphdata = mostgraph_total_data_numpy.shape[0]
# extract test data
mostgraph_abnormal_test_data_numpy = mostgraph_total_data_numpy[0:number_of_test]
# extract train and validation data
mostgraph_abnormal_train_data_numpy = mostgraph_total_data_numpy[number_of_test:number_of_abnormal_mostgraphdata]

# combine respiratory normal and asthmatics subjects of train and validation data
mostgraph_train_data_numpy = np.concatenate([mostgraph_normal_train_data_numpy, mostgraph_abnormal_train_data_numpy])

# split label and train data
labels_train, data_train = np.split(mostgraph_train_data_numpy, [1], axis=1)

# Correct for data imbalance by oversampling using SMOTE
from imblearn.over_sampling import SMOTE #, KMeansSMOTE, SVMSMOTE, ADASYN
sm = SMOTE() #you can select other oversampling methods like ADASYN, KMeansSMOTE()
# for downsampling, use following 2 instructions instead
#  from imblearn.under_sampling import RandomUnderSampler
#  sm = RandomUnderSampler()
x_resampled, y_resampled = sm.fit_resample(data_train, labels_train)
print(x_resampled.shape)
print(y_resampled[..., np.newaxis].shape)

print(y_resampled)
print(x_resampled)

# Set trainig data
X_train = x_resampled
# Objective variable as "diagnosis" 
y_train = np.reshape(y_resampled,(-1)) # [[]] -> []

# LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=10000)
model = LogisticRegression()
model.fit(X_train,y_train)
print("model coefficient", model.coef_)
print("model intercept", model.intercept_)

# test the model
# normal control
labels_test, data_test = np.split(mostgraph_normal_test_data_numpy, [1], axis=1)
X_test = data_test
y_test = np.reshape(labels_test,(-1)) # [[]] -> []
predicted1 = model.predict_proba(X_test)
#print("y_score for normal", predicted)
groundtruth1 = y_test

# test the model
# asthmatics
labels_test, data_test = np.split(mostgraph_abnormal_test_data_numpy, [1], axis=1)
X_test = data_test
y_test = np.reshape(labels_test,(-1)) # [[]] -> []
predicted2 = model.predict_proba(X_test)
groundtruth2 = y_test

y_true = np.concatenate([groundtruth1, groundtruth2])
y_predict = np.concatenate([predicted1[:, 1], predicted2[:, 1]])

# Draw ROC
from sklearn.metrics import RocCurveDisplay
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay.from_predictions
RocCurveDisplay.from_predictions(
    y_true,
    y_predict,
    name= "normal vs asthmatics",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("normal vs asthmatics")
plt.legend()
plt.savefig('./roc_logistic.jpg', dpi=2400)
plt.show()

# Compute the area under the ROC curve.
from sklearn.metrics import roc_auc_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
auc = roc_auc_score(y_true, y_predict)
print("AUC {:.2f}".format(auc))


# Youden index
from sklearn.metrics import roc_curve
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
fpr, tpr, thresholds = roc_curve(y_true, y_predict)
index = np.argmax(tpr - fpr)
cutoff = thresholds[index]
sensitivity = tpr[index]
specificity = 1 - fpr[index]
acc = (sensitivity + specificity) / 2
prec = sensitivity / (sensitivity + fpr[index])
f1 = 2 * prec * sensitivity / (prec + sensitivity)
print("Threshould:", cutoff,"Sensitivity(Recall):", sensitivity, "Specificity：", specificity, "Accuracy：", acc, "Precision：", prec,"F1 score：", f1)
