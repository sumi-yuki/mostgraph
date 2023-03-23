#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# output file name
HeatmapFileName = "male_heatmap.csv"
# reads reference ranges
ReferenceFileName = "male_reference_range.csv"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#List of measured Item
MeasuredItemList = ["R5", "R5in", "R5ex", "R5delta", "R20", "R20in", "R20ex", "R20delta", "R5-R20", "R5-R20in", "R5-R20ex", "R5-R20delta", "X5", "X5in", "X5ex", "X5delta", "Fres", "Fresin", "Fresex", "Fresdelta", "ALX", "ALXin", "ALXex", "ALXdelta"]

# open file, write Header
f = open(HeatmapFileName, 'w')
print(','.join(MeasuredItemList + ["diagnosis"]), flush=True, file=f)

# Read the reference values (csv file)
ReferenceData_pandas = pd.read_csv(ReferenceFileName, index_col=0, encoding="utf-8")

# reads Mostgraph measured data from TargetFileName, judge, and returns JudgedValues_pandas　as 0(normal) or 1(disease)
def judge_abnormal(TargetFileName, DiseaseLabel):
    # Read the target file to evaluate (csv file)
    MostgraphData_pandas = pd.read_csv(TargetFileName, encoding="utf-8")
    #Define the valuable to save Results i.e.:0(normal) or 1(abnormal)
    JudgedValues_pandas = pd.DataFrame() #empty pandas Dataframe
    #Process every measured item
    for MeasuredItem in MeasuredItemList:    # print(MeasuredItem) # measured items
        lower =  ReferenceData_pandas.loc[[MeasuredItem], "lower"].values.squeeze()   # [[]] -> []
        upper =  ReferenceData_pandas.loc[[MeasuredItem], "upper"].values.squeeze()
        # Select measured item (example R5) data to judge
        MeasuredValues_pandas = MostgraphData_pandas.loc[:,[MeasuredItem]]
        # Convert to numpy
        MeasuredValues_numpy = MeasuredValues_pandas.values.squeeze()   # [[]] -> []
        #Within normal range (0) or not (1)?
        JudgedValues_numpy   = list(map(lambda x: 0 if( x > lower and x < upper) else 1, MeasuredValues_numpy))
        #Store the results
        JudgedValues_pandas[MeasuredItem] = JudgedValues_numpy
        #end of Item loop

    JudgedValuesT_pandas = JudgedValues_pandas.T
    # Display heatmap
    # sns.heatmap(JudgedValuesT_pandas, xticklabels="patients", cbar=False) # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    ax = sns.heatmap(JudgedValuesT_pandas, cbar=False) # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    ax.set_title(DiseaseLabel)
    plt.savefig("maleHeatmap"+DiseaseLabel+".jpg", dpi=1200)
    plt.show()
    ax = sns.clustermap(JudgedValuesT_pandas, cbar_pos=None) #https://seaborn.pydata.org/generated/seaborn.clustermap.html
    #ax.ax_col_dendrogram.set_title(DiseaseLabel)
    plt.savefig("maleClusterHeatmap"+DiseaseLabel+".jpg", dpi=2400)
    plt.show()
    return JudgedValues_pandas

# Examine if the number of abnormal values can discriminate between normal and abnormal
# NORMAL
# return as 0(normal) or 1(disease)
JudgedValues_pandas = judge_abnormal("male_control.csv", "Male Normal")
# Convert to numpy
JudgedValues_numpy = JudgedValues_pandas.to_numpy()
# Count the number of abnormal items
NumberOfAbnormalItems_in_Normal = np.sum(JudgedValues_numpy, axis=1)
#append to HeatmapFileName
JudgedValues_pandas["diagnosis"] = 0
print(JudgedValues_pandas)
JudgedValues_pandas.to_csv(HeatmapFileName, mode='a', header=False, index=False)


# Examine if the number of abnormal values can discriminate between normal and abnormal
# ASTHMA
# return as 0(normal) or 1(disease)
JudgedValues_pandas = judge_abnormal("male_ba.csv", "Male Asthma")
# Convert to numpy
JudgedValues_numpy = JudgedValues_pandas.to_numpy()
# Count the number of abnormal items
NumberOfAbnormalItems_in_Asthma = np.sum(JudgedValues_numpy, axis=1)
#append to HeatmapFileName
JudgedValues_pandas["diagnosis"] = 1
print(JudgedValues_pandas)
JudgedValues_pandas.to_csv(HeatmapFileName, mode='a', header=False, index=False)

# close output file
f.close()

#ROC curve
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
      
plt.title('ROC curve in Male')
plt.plot(FalsePositiveRate, TruePositiveRate, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.savefig("maleROC.jpg", dpi=1200)
plt.show()

