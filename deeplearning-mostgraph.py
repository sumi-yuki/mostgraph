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

# tensorflow https://www.tensorflow.org/?hl=en
# To install,
# pip install tensorflow==2.10.0

# tensorflow.js https://www.tensorflow.org/js?hl=en
# To install,
# pip install tensorflowjs==3.21.0

# Matplotlib https://matplotlib.org
# To install,
# pip install matplotlib
# conda install -c conda-forge matplotlib

# scikit-learn https://scikit-learn.org/stable/
# To install,
# pip install -U scikit-learn 
#  or
# conda install -c conda-forge matplotlib

# imbalanced-learn https://imbalanced-learn.org/stable/index.html
# To install,
# pip install -U imbalanced-learn
#  or
# conda install -c conda-forge imbalanced-learn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential

# the number of test data for the metrics
number_of_test = 10

# read the Mostgraph measurement results of respiratory normal subjects
# Column of csv file consist of "diagnosis", "gender", "R5in", "R5ex", "R20in", "R20ex", "X5in", "X5ex", "Fresin", "Fresex", "ALXin", "ALXex"
mostgraphdata = pd.read_csv("female,male_controldata_smart.csv", encoding="utf-8")
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
# apply oversampling to the respiratory normal subject
mostgraph_normal_train_data_numpy_extended=np.tile(mostgraph_normal_train_data_numpy,(5, 1))

# read the Mostgraph measurement results of asthmatics subjects
mostgraphdata = pd.read_csv("female,male_patientsdata_smart.csv", encoding="utf-8")
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
mostgraph_train_data_numpy = np.concatenate([mostgraph_normal_train_data_numpy_extended, mostgraph_abnormal_train_data_numpy])
# shuffle to make sequence random
np.random.shuffle(mostgraph_train_data_numpy)

#Correct for data imbalance
number_of_normal_mostgraphdata_extended = mostgraph_normal_train_data_numpy_extended.shape[0]
number_of_normal_mostgraphdata = number_of_normal_mostgraphdata_extended - number_of_test
number_of_abnormal_mostgraphdata = number_of_abnormal_mostgraphdata - number_of_test
initial_bias = np.log([number_of_abnormal_mostgraphdata / number_of_normal_mostgraphdata])
print("Initial bias: {:.5f}".format(initial_bias[0]))
number_of_total_train_mostgraphdata = number_of_normal_mostgraphdata + number_of_abnormal_mostgraphdata
weight_for_0 = (1 / number_of_normal_mostgraphdata) * (number_of_total_train_mostgraphdata) / 2.0
weight_for_1 = (1 / number_of_abnormal_mostgraphdata) * (number_of_total_train_mostgraphdata) / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))

# "diagnosis"
labels_train = mostgraph_train_data_numpy[ : , 0:1]
# "gender", "R5in", "R5ex", "R20in", "R20ex", "X5in", "X5ex", "Fresin", "Fresex", "ALXin", "ALXex"
data_train = mostgraph_train_data_numpy[ : , 1:12]

# define deep learning model
model = Sequential()
model.add(Dense(units=128, input_shape=(11,)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(units=64))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(units=32))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(units=16))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=["accuracy"])
model.summary() 

# train the model
model_history = model.fit(data_train, labels_train, class_weight=class_weight, batch_size = 32, epochs = 2500, validation_split = 0.05)

# display trainig statistics
history_dict = model_history.history
# load Loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
# load Accaracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
# make list from 1 to epoch
epochlist = range(1, len(loss_values) +1)
# write graph
plt.plot(epochlist, acc, 'go', label='Accuracy at training')
plt.plot(epochlist, val_acc, 'b', label='Accuracy at validation')
plt.plot(epochlist, loss_values, 'mo', label='Loss at training')
plt.plot(epochlist, val_loss_values, 'r', label='Loss at validation')
# title
plt.title('Training and Validation')
plt.xlabel('Epochs')
plt.legend()
# display and save
plt.savefig("Training.jpg", dpi=2400)
plt.show()


# test the model for 
labels_test = mostgraph_normal_test_data_numpy[ : , 0:1]
data_test = mostgraph_normal_test_data_numpy[ : , 1:14]
loss, accuracy = model.evaluate(data_test, labels_test)
print("Accuracy for normal= {:.2f}".format(accuracy))
predicted = model.predict(data_test)
print("predicted for normal:", predicted)
#print(data_test)

labels_test = mostgraph_abnormal_test_data_numpy[ : , 0:1]
data_test = mostgraph_abnormal_test_data_numpy[ : , 1:14]
loss, accuracy = model.evaluate(data_test, labels_test)
print("Accuracy for abnormal= {:.2f}".format(accuracy))
predicted = model.predict(data_test)
print("predicted for abnormal:", predicted)
#print(data_test)

# save the trained model for python
model.save("deeplearing_model")
# save the trained model for javascript
#import tensorflowjs as tfjs
#tfjs.converters.save_keras_model(model, "./mostgraph_model")
