#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# seaborn https://seaborn.pydata.org/installing.html
# To install,
# pip install seaborn
#  or
# conda install seaborn -c conda-forge

# scikit-learn https://scikit-learn.org/stable/
# To install,
# pip install -U scikit-learn 
#  or
# conda install -c conda-forge matplotlib

# Matplotlib https://matplotlib.org
# To install,
# pip install matplotlib
#  or
# conda install -c conda-forge matplotlib

# gender
gender = "Female"

# File name of "reference ranges"
# row (measured item name, lower-limit, estimated value, upper-limit)
# column with header (R5, R5in, R5ex, R5delta, R20, R20in, R20ex, R20delta, R5-R20, R5-R20in, R5-R20ex, R5-R20delta, X5, X5in, X5ex, X5delta, Fres, Fresin, Fresex, Fresdelta, ALX, ALXin, ALXex, ALXdelta)
ReferenceFileName = "female_reference_range.csv"

# input FOT measured file name for normal subjects
MeasuredValueFileNameNormal = "female_control.csv"
# input FOT measured file name for asthmatic subjects
MeasuredValueFileNameAsthmatics = "female_ba.csv"

# output heatmap csv file name
# row measured item (R5, R5in, R5ex, R5delta, R20, R20in, R20ex, R20delta, R5-R20, R5-R20in, R5-R20ex, R5-R20delta, X5, X5in, X5ex, X5delta, Fres, Fresin, Fresex, Fresdelta, ALX, ALXin, ALXex, ALXdelta)
# column each patient with header, indicating each measured item 0:within reference range, 1:out of refeerence range  
HeatmapFileName = "female_heatmap.csv"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# List of measured Item
MeasuredItemList = ["R5", "R5in", "R5ex", "R5delta", "R20", "R20in", "R20ex", "R20delta", "R5-R20", "R5-R20in", "R5-R20ex", "R5-R20delta", "X5", "X5in", "X5ex", "X5delta", "Fres", "Fresin", "Fresex", "Fresdelta", "ALX", "ALXin", "ALXex", "ALXdelta"]

# open heatmap file to output, write Header
f = open(HeatmapFileName, 'w')
print(','.join(MeasuredItemList + ["diagnosis"]), flush=True, file=f)

# Read the reference ranges (csv file)
# row (measured item name, lower-limit, expected mean value, upper-limit)
# column with header (R5, R5in, R5ex, R5delta, R20, R20in, R20ex, R20delta, R5-R20, R5-R20in, R5-R20ex, R5-R20delta, X5, X5in, X5ex, X5delta, Fres, Fresin, Fresex, Fresdelta, ALX, ALXin, ALXex, ALXdelta)
ReferenceData_pandas = pd.read_csv(ReferenceFileName, index_col=0, encoding="utf-8")

# reads Mostgraph measured data from TargetFileName, judge if each measured value is within reference range, and returns JudgedValues_pandas　as 0(normal) or 1(disease)
def judge_abnormal(TargetFileName, DiseaseLabel):
    # Input:
    #  TargetFileName: csv file name
    #   row (age	Ht	BW, R5, R5in, R5ex, R5delta, R20, R20in, R20ex, R20delta, R5-R20, R5-R20in, R5-R20ex, R5-R20delta, X5, X5in, X5ex, X5delta, Fres, Fresin, Fresex, Fresdelta, ALX, ALXin, ALXex, ALXdelta)
    #   column with header (Subjects [normal control or asthmatics])
    #  DiseaseLabel: label to be printed
    # Output file:
    #  heatmap figure as DiseaseLabel+"Heatmap.jpg"
    #  cluster figure as DiseaseLabel+"ClusterHeatmap.jpg"  
    # Return:
    #  Table in Pandas
    #   row measured item + diagnosis (R5, R5in, R5ex, R5delta, R20, R20in, R20ex, R20delta, R5-R20, R5-R20in, R5-R20ex, R5-R20delta, X5, X5in, X5ex, X5delta, Fres, Fresin, Fresex, Fresdelta, ALX, ALXin, ALXex, ALXdelta, diagnosis)
    #   column each patient with header, indicating each measured item 0:within reference range, 1:out of refeerence range  

    # Read the target file to evaluate (csv file)
    MostgraphData_pandas = pd.read_csv(TargetFileName, encoding="utf-8")
    # Define the valuable to save Results i.e.:0(normal) or 1(abnormal)
    JudgedValues_pandas = pd.DataFrame() #empty pandas Dataframe
    # Process every measured item
    for MeasuredItem in MeasuredItemList:    # print(MeasuredItem) # measured items
        lowerlimit =  ReferenceData_pandas.loc[[MeasuredItem], "lower"].values.squeeze()   # [[]] -> []
        upperlimit =  ReferenceData_pandas.loc[[MeasuredItem], "upper"].values.squeeze()
        # Select measured item (example R5) data to judge
        MeasuredValues_pandas = MostgraphData_pandas.loc[:,[MeasuredItem]]
        # Convert to numpy
        MeasuredValues_numpy = MeasuredValues_pandas.values.squeeze()   # [[]] -> []
        #Within normal range (0) or not (1)?
        JudgedValues_numpy   = list(map(lambda x: 0 if( x > lowerlimit and x < upperlimit) else 1, MeasuredValues_numpy))
        #Store the results
        JudgedValues_pandas[MeasuredItem] = JudgedValues_numpy
        #end of Item loop

    JudgedValuesT_pandas = JudgedValues_pandas.T
    # Display heatmap
    ax = sns.heatmap(JudgedValuesT_pandas, cbar=False) # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    ax.set_title(DiseaseLabel)
    plt.savefig(DiseaseLabel+"Heatmap.jpg", dpi=1200)
    plt.show()
    # Display heatmap with cluster analysis
    ax = sns.clustermap(JudgedValuesT_pandas, cbar_pos=None) #https://seaborn.pydata.org/generated/seaborn.clustermap.html
    plt.savefig(DiseaseLabel+"ClusterHeatmap.jpg", dpi=2400)
    plt.show()
    return JudgedValues_pandas

# Examine if the number of abnormal values can discriminate between normal and abnormal
# NORMAL
# return as 0(normal) or 1(disease)
JudgedValues_pandas = judge_abnormal(MeasuredValueFileNameNormal, gender+"Normal")
# Convert to numpy
JudgedValues_numpy = JudgedValues_pandas.to_numpy()
# Count the number of abnormal items
NumberOfAbnormalItems_in_Normal = np.sum(JudgedValues_numpy, axis=1)
# append to Heatmap File
JudgedValues_pandas["diagnosis"] = 0
print(JudgedValues_pandas)
JudgedValues_pandas.to_csv(HeatmapFileName, mode='a', header=False, index=False)
# Examine if the number of abnormal values can discriminate between normal and abnormal
# ASTHMA
# return as 0(normal) or 1(disease)
JudgedValues_pandas = judge_abnormal(MeasuredValueFileNameAsthmatics, gender+"Asthma")
# Convert to numpy
JudgedValues_numpy = JudgedValues_pandas.to_numpy()
# Count the number of abnormal items
NumberOfAbnormalItems_in_Asthma = np.sum(JudgedValues_numpy, axis=1)
# append to Heatmap File
JudgedValues_pandas["diagnosis"] = 1
print(JudgedValues_pandas)
JudgedValues_pandas.to_csv(HeatmapFileName, mode='a', header=False, index=False)
# close output heatmap csv file
f.close()

# Draw ROC curve
TotalNumber_in_Normal = len(NumberOfAbnormalItems_in_Normal)
TotalNumber_in_Asthma = len(NumberOfAbnormalItems_in_Asthma)
FalsePositiveRate = [] # make null list
TruePositiveRate = [] # make null list
for threshould in range(len(MeasuredItemList)):
    AbnormalNumber_in_Normal = np.count_nonzero(NumberOfAbnormalItems_in_Normal >= threshould)
    AbnormalNumber_in_Asthma = np.count_nonzero(NumberOfAbnormalItems_in_Asthma >= threshould)
    FalsePositiveRate.append(AbnormalNumber_in_Normal/TotalNumber_in_Normal)
    TruePositiveRate.append(AbnormalNumber_in_Asthma/TotalNumber_in_Asthma)
    #statistic
    #print("threshould", threshould, "TotalNumber_in_Normal", TotalNumber_in_Normal, "AbnormalNumber_in_Normal", AbnormalNumber_in_Normal, "TotalNumber_in_Asthma", TotalNumber_in_Asthma, "AbnormalNumber_in_Asthma", AbnormalNumber_in_Asthma)
    print("threshould", threshould, "Accauacy_in_Normal", (TotalNumber_in_Normal-AbnormalNumber_in_Normal)/TotalNumber_in_Normal, "Accauacy_in_Asthma", AbnormalNumber_in_Asthma / TotalNumber_in_Asthma)
from sklearn import metrics
roc_auc = metrics.auc(FalsePositiveRate, TruePositiveRate)
print("AUC:",roc_auc)
plt.figure(figsize=(6,6))       
plt.title('ROC curve by the number of items out of rerference range')
plt.axis([0, 1, 0, 1]) 
plt.plot(FalsePositiveRate, TruePositiveRate, marker='o', label="normal vs asthmatics")
plt.plot([0, 1], [0, 1], linestyle='dashed', color='black', label="Chance level")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.savefig(gender+"ROC.jpg", dpi=1200)
plt.show()
