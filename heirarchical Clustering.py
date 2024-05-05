# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 09:45:31 2023

@author: H P
"""
# HIERARCHICAL CLUSTERING ASSIGNMENTS

###############ASSIGNMENT NO:1######################

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel(r"C:\Users\H P\OneDrive\Documents\clustering\Telco_customer_churn.xlsx")
data
data.drop(["Count"],axis = 1, inplace = True)
data.drop(["Quarter"],axis = 1, inplace = True)


import numpy as np
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
for column in numerical_columns:
    plt.boxplot(data[column])
    plt.title("Boxplot of "+column)
    plt.show()
    
    IQR = data[column].quantile(0.75) - data[column].quantile(0.25)
    lower_limit = data[column].quantile(0.25) - (1.5 * IQR)
    upper_limit = data[column].quantile(0.75) + (1.5 * IQR)
    data[column] = np.clip(data[column], lower_limit, upper_limit)
    
    plt.boxplot(data[column])   
    plt.title("New plot of "+column)
    plt.show()

#mode imputer
from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
data["Offer"] = pd.DataFrame(median_imputer.fit_transform(data[["Offer"]]))
data["Offer"].isna().sum()

median_imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
data["Internet_Type"] = pd.DataFrame(median_imputer.fit_transform(data[["Internet_Type"]]))
data["Internet_Type"].isna().sum()

data.isna().sum()

data.drop(["Customer_ID","Total_Refunds","Total_Extra_Data_Charges"],axis = 1,inplace = True)

#DUPLICATE
duplicate = data.duplicated()
sum(duplicate)
#COVERTING CATEGARICAL TO NEWMERICAL
data2 = pd.get_dummies(data, drop_first = True)
new_data = data2.astype(int)
#NORMALIZATION 
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x
# Assuming new_data1 is the dataset you want to normalize
norm_data = norm_func(new_data)
#CLUSTERING
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(norm_data,method = "complete", metric = "euclidean")
#dendogram
plt.figure(figsize=(15, 8))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")

# Assuming `z` contains your linkage matrix or distance matrixs
sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)

plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = "complete", affinity = "euclidean").fit(norm_data)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
new_data["Cluster"] = cluster_labels
new_data.shape
new_data.columns
data_1 = new_data.iloc[:,[31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]] ####
#Aggregate mean of each cluster
new_data.groupby(new_data['Cluster']).mean()
#Aggregate mean of each cluster
new_data.iloc[:,1:].groupby(new_data.Cluster).mean()

###################################ASSINGMENT NO:2####################################
import pandas as pd
data = pd.read_csv(r"C:\Users\H P\OneDrive\Documents\clustering\AutoInsurance.csv")
data
data.columns
data.Customer.unique()
data.State.unique()
data.drop(["Customer"],axis=1,inplace=True)

import numpy as np
import matplotlib.pyplot as plt

numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
for column in numerical_columns:
    plt.boxplot(data[column])
    plt.title("Boxplot of "+column)
    plt.show()
    
    IQR = data[column].quantile(0.75) - data[column].quantile(0.25)
    lower_limit = data[column].quantile(0.25) - (1.5 * IQR)
    upper_limit = data[column].quantile(0.75) + (1.5 * IQR)
    data[column] = np.clip(data[column], lower_limit, upper_limit)
    
    plt.boxplot(data[column])   
    plt.title("New plot of "+column)
    plt.show()

data.Customer_Lifetim_Value.unique()
data.Number_of_Open_Complaints.unique()
data.Total_Claim_Amount.unique()
data.drop(["Number_of_Open_Complaints"],axis=1,inplace=True)
#MISSING VALUES
data.isna().sum()
#DUPLICATE VALUE
duplicate = data.duplicated()
duplicate
sum(duplicate)
data_6 = data.drop_duplicates()
#DUMMY VARRIABLE
data_new = pd.get_dummies(data_6, drop_first = True)
new_data = data_new.astype(int)
new_data.isna().sum()
data_new.shape
#NORMALIZATION METHOD
def norm_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return(x)
new_data1 = norm_func(new_data)
#CLUSTERING
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(new_data1,method = "complete", metric = "euclidean")

#DENDROGRAM
plt.figure(figsize = (15,8));plt.title("Hierarchical Clustering Dendrogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(z,
               leaf_rotation = 0,
               leaf_font_size = 10
               )
plt.show()
#AGGLOMERATIVE CLUSTERING USING 3
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = "complete", affinity = "euclidean").fit(new_data1)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
data["Cluster"] = cluster_labels
#CHANGING COLUMN POSITION
data.shape
data_1 = data.iloc[:,[22,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
#AGGREGATIVE MEAN OF EACH CLUSTER
data.iloc[:,].groupby(data.Cluster).mean()

#############ASSIGNMENT NO:3######################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv(r"C:\Users\H P\OneDrive\Documents\clustering\crime_data.csv")
data
data.columns
data.place.unique()
data.drop(["place"], axis = 1, inplace = True)
#MISSING VALUE IMPUTATION
data.isna().sum()
#OUTLIERS CHECKING METHODE
columns_list = ["Murder","Assault","UrbanPop","Rape"]
for columns in columns_list:
    plt.boxplot(data[columns])
    plt.title("boxplot of " +columns)
    plt.xlabel(columns)
    plt.show()
    
#OUTLIER REMOVING METHOD USING WINSORIZER
from feature_engine.outliers import Winsorizer
winsorizer = Winsorizer(capping_method = 'iqr',
                        tail = 'right',
                        fold = 1.5,
                        variables = ["Rape"])
data["Rape"] = winsorizer.fit_transform(data[["Rape"]])
    
#DUPLICATE VALUES
duplicate = data.duplicated()
sum(duplicate)

#NORMALIZATION
def norm_func(i):
    x = (i+i.min()) / (i.max()-i.min())
    return(x)
norm_data = norm_func(data)

#CLUSTERING DATA
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(norm_data, method = "complete",metric = "euclidean")

#dendrogram
plt.figure(figsize=(15,8));plt.title("Hierarchical Clustering Dendrogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(z,
               leaf_rotation = 0,
               leaf_font_size = 10)
#Agglomerative Clustering using 3
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage ="complete", affinity = "euclidean").fit(norm_data)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
data["Clust"] = cluster_labels
new_data = data.iloc[:,[4,0,1,2,3]]

#Aggregate mean of each cluster
new_data.iloc[:,1:].groupby(data.Clust).mean()


##################ASSIGNMENT NO :4#######################
import pandas as pd
data = pd.read_excel(r"C:\Users\H P\OneDrive\Documents\clustering\EastWestAirlines.xlsx")
data
data.columns

data.cc1_miles.unique()
data.cc2_miles.unique()
data.cc3_miles.unique()
#OUTLIER METHOD
import matplotlib.pyplot as plt
import numpy as np
columns_list = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll', 'Award?']
for columns in columns_list:
    plt.boxplot(data[columns])
    plt.title("Boxplot of "+columns)
    plt.xlabel(columns)
    plt.show()
    
#REMOVING OUTLIERS
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
for column in numerical_columns:
    plt.boxplot(data[column])
    plt.title("Boxplot of "+column)
    plt.show()
    
    IQR = data[column].quantile(0.75) - data[column].quantile(0.25)
    lower_limit = data[column].quantile(0.25) - (1.5 * IQR)
    upper_limit = data[column].quantile(0.75) + (1.5 * IQR)
    data[column] = np.clip(data[column], lower_limit, upper_limit)
    
    plt.boxplot(data[column])   
    plt.title("New plot of "+column)
    plt.show()
    
data.Balance.unique()
data.Qual_miles.unique()
data.cc2_miles.unique()
data.cc3_miles.unique()
data.drop(['Qual_miles','cc2_miles','cc3_miles','ID#'],axis = 1,inplace = True)

#MISSING VALUE
data.isna().sum()
#DUPLICATE VALUE
duplicate = data.duplicated()
sum(duplicate)
data_1 = data.drop_duplicates()
#NORMALIZATION
def norm_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return(x)
norm_data = norm_func(data_1)
#CLUSTERING
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(norm_data, method = "complete",metric = "euclidean")

#Dendrogram
plt.figure(figsize=(15,8));plt.title("Hierarchical Clustering Dendrogram");plt.xlabel("Index");plt.ylabel("distance")
sch.dendrogram(z,
               leaf_rotation = 0,
               leaf_font_size = 10)
plt.show()
#Agglomerative clustering choosing 3
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = "complete", affinity = "euclidean").fit(norm_data)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
data["Cluster"] = cluster_labels
#Column position change
data.shape
new_data = data.iloc[:,[8,0,1,2,3,4,5,6,7]]
#Aggregate mean of each cluster
new_data.iloc[:,1:].groupby(new_data.Cluster).mean()

