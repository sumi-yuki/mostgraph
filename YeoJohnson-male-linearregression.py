#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# install modules

# scikit-learn https://scikit-learn.org/stable/
# pip install -U scikit-learn
#  or
# conda install -c conda-forge matplotlib

# Matplotlib https://matplotlib.org
# pip install matplotlib
#  or
# conda install -c conda-forge matplotlib

import numpy as np
import pandas as pd
import statistics
from sklearn import linear_model
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

# List of Mostgraph measured item
MeasuredItemList = ("R5", "R5in", "R5ex", "R5delta", "R20", "R20in", "R20ex", "R20delta", "R5-R20", "R5-R20in", "R5-R20ex", "R5-R20delta", "X5", "X5in", "X5ex", "X5delta", "Fres", "Fresin", "Fresex", "Fresdelta", "ALX", "ALXin", "ALXex", "ALXdelta")
MostgraphData_pandas = pd.read_csv("male_control.csv", encoding="utf-8")

# List of explanatory variables
IndexItemList = ("age", "Ht", "BW")
# Read (extract) explanatory variables
IndexValues_pandas = MostgraphData_pandas.loc[:,IndexItemList]
IndexValues_numpy = IndexValues_pandas.values

for MeasuredItem in MeasuredItemList:
    # Extract colume of each measured item example R5
    MeasuredValues_pandas = MostgraphData_pandas.loc[:,[MeasuredItem]]   
    # convert to numpy from pandas dataframe
    MeasuredValues_numpy = MeasuredValues_pandas.values

    # Transformation
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    # Box-Cox tranformation
    # pt = PowerTransformer(method='box-cox')
    # Yeo-Johnson transformation
    pt = PowerTransformer(method='yeo-johnson')
    pt.fit(MeasuredValues_numpy)
    TransformedValues_numpy = pt.transform(MeasuredValues_numpy)   # [[]]
    
    # Statistics after transformation
    SimpleMean = statistics.mean(TransformedValues_numpy.squeeze()) # [[]] -> []
    SimpleStd = statistics.stdev(TransformedValues_numpy.squeeze()) # [[]] -> []
    
    # Inverse transformation of mean±2SD
    UpperLimit = pt.inverse_transform(np.array([[SimpleMean + 2 * SimpleStd]])).squeeze()  #  -> [[]] -> [[]] ->
    EstimatedValue = pt.inverse_transform(np.array([[SimpleMean]])).squeeze()  #  -> [[]] -> [[]] ->
    LowerLimit = pt.inverse_transform(np.array([[SimpleMean - 2 * SimpleStd]])).squeeze()  #  -> [[]] -> [[]] ->
    print(f'{MeasuredItem}',f'{LowerLimit:.2f} - {UpperLimit:.2f}', sep='\t') 

    # Multiple regression
    MultipleLinearRegression = linear_model.LinearRegression()
    MultipleLinearRegression.fit(IndexValues_numpy, TransformedValues_numpy)
    PredictedTransformedValues = MultipleLinearRegression.predict(IndexValues_numpy)
    diff = PredictedTransformedValues - TransformedValues_numpy
    PredictedOriginalValues = pt.inverse_transform(PredictedTransformedValues)
    
    # Statistics of prediction
    DiffMean = statistics.mean(diff.squeeze()) # [[]] -> []
    DiffStd = statistics.stdev(diff.squeeze()) # [[]] -> []
    
    # original versus predicted
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
    fig.suptitle("Male: Predicted vs Measured " + MeasuredItem, size="xx-large")
     # Transforned measured item
    axs[0].set_xlabel("Measured transformed " + MeasuredItem)
    axs[0].set_ylabel("Predicteded transformed " + MeasuredItem)
    axs[0].set_title("Yeo-Johnson transformed Values", color="green")
    axs[0].vlines(TransformedValues_numpy,PredictedTransformedValues-2*DiffStd, PredictedTransformedValues+2*DiffStd, color="green", linewidth=1)
    axs[0].axhspan(ymin=SimpleMean-2*SimpleStd, ymax=SimpleMean+2*SimpleStd, xmin=0, xmax=1, color="red", alpha=0.1)
    axs[0].scatter(TransformedValues_numpy, PredictedTransformedValues, c='black')
    axs[0].plot([-4,4], [-4,4], c='black')
     # Original measured item
    axs[1].set_xlabel("Measured " + MeasuredItem)
    axs[1].set_ylabel("Predicteded " + MeasuredItem)
    axs[1].set_title("Original measured values",  color="blue")
    axs[1].axhspan(ymin=LowerLimit, ymax=UpperLimit, xmin=0, xmax=1, color="red", alpha=0.1)
    axs[1].vlines(MeasuredValues_numpy,pt.inverse_transform(PredictedTransformedValues-2*DiffStd), pt.inverse_transform(PredictedTransformedValues+2*DiffStd), color="blue", linewidth=1)
    axs[1].scatter(MeasuredValues_numpy, PredictedOriginalValues, c='black')
    axs[1].plot([-5,20], [-5,20], c='black')
    plt.savefig("maleOriginal-Predicted" + MeasuredItem + ".jpg", dpi=2400)
    plt.show()
