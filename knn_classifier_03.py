# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
This code takes the data from csv file and calls kNN_classifier 
and measures accuracy. 
"""
import knn_utils as kn
from sklearn.neighbors import KNeighborsClassifier

data_file_name = 'data/student_marks_record.csv'
df = kn.csv_read(data_file_name)

df_data = df[["maths","science", "computer"]].values
df_labels =  df["category"].values

print(df[["maths","science", "computer"]].values)
print(df["category"].values)

train_data = df_data[0:10]
train_labels =df_labels[0:10]
test_data = df_data[10:15]
test_labels =df_labels[10:15]

kNN = 7
test_prediction =[]

knn_classifier = KNeighborsClassifier(n_neighbors = kNN)
knn_classifier.fit(train_data,train_labels)

for test_fv in test_data:    
    test_prediction.append(knn_classifier.predict([test_fv]))
   
    
    
print("test prediction classes are :", test_prediction)

accuracy = kn.accuracy_cal(test_labels,test_prediction)

print ("Test Accuracy (in percentage):", accuracy)