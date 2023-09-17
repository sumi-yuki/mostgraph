#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# install modules

# Matplotlib https://matplotlib.org
# To install,
# pip install matplotlib
# conda install -c conda-forge matplotlib

# scikit-learn https://scikit-learn.org/stable/
# To install,
# pip install -U scikit-learn 
#  or
# conda install -c conda-forge matplotlib

SavedFileName = "female_reference_range.csv"
# open file, write Header
f = open(SavedFileName, 'w')
print('Item,lower,estimated,upper', file=f)

import numpy as np
import pandas as pd
import statistics
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

# List of Mostgraph measured item
MeasuredItemList = ("R5", "R5in", "R5ex", "R5delta", "R20", "R20in", "R20ex", "R20delta", "R5-R20", "R5-R20in", "R5-R20ex", "R5-R20delta", "X5", "X5in", "X5ex", "X5delta", "Fres", "Fresin", "Fresex", "Fresdelta", "ALX", "ALXin", "ALXex", "ALXdelta")
MostgraphData_pandas = pd.read_csv("female_control.csv", encoding="utf-8")

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
    print(f'{MeasuredItem},{LowerLimit},{EstimatedValue},{UpperLimit}', file=f) 

    # Histogram https://seaborn.pydata.org/generated/seaborn.histplot.html
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
    fig.suptitle("Female: " + MeasuredItem + " Histogram", size="xx-large")
    axs[0].hist(MeasuredValues_numpy, facecolor="blue")
    axs[0].set_xlabel(MeasuredItem)
    axs[0].set_ylabel("count")
    axs[0].set_title("a   Measured values", color="black")
    axs[1].hist(TransformedValues_numpy, facecolor="green")
    axs[1].set_xlabel("Yeo-Johnson transformed " + MeasuredItem)
    axs[1].set_ylabel("count")
    axs[1].set_title("b   Yeo-Johnson transformed λ=" + str(pt.lambdas_), color="black")
    axs[1].vlines(SimpleMean,0,30,color="red", linewidth=10, linestyle='dotted')
    axs[1].axvspan(xmin=SimpleMean-2*SimpleStd, xmax=SimpleMean+2*SimpleStd, ymin=0, ymax=1, color="red", alpha=0.1)
    plt.savefig("FemaleHistogram " + MeasuredItem + ".jpg", dpi=1200)
    plt.show()

f.close()
