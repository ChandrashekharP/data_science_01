# -*- coding: utf-8 -*-
"""
Created on Thu May 13 22:59:38 2021

@author: IRDC Lab
"""
import pandas as pd
import numpy as np 

def knn_class_pred(kNN,test_fv, train_data, train_labels):
    unique_labels = np.unique(train_labels)
    distance= []
    for train_fv in train_data:
        dist = [(a-b)**2 for a,b in zip(test_fv, train_fv)]
        distance.append(sum(dist)/len(train_labels))
    distance = np.array(distance)
    sort_inds= np.argsort(distance)
    nn_classes = train_labels[sort_inds[0:kNN]]
    class_count = np.array([0 for a in range(len(unique_labels))])
    for nn in nn_classes:
        for i,label in enumerate(unique_labels):
            if  nn == label:
                class_count[i] += 1
    sort_i= np.argsort(class_count)            
    prediction_class = unique_labels[sort_i[len(unique_labels)-1]]    
    print(test_fv, prediction_class )
    
    return prediction_class


def csv_read(csv_file):
    data = pd.read_csv(csv_file) 
    return data

def accuracy_cal(test_labels,test_prediction):    
    correct_count = 0
    for a,b in zip(test_labels, test_prediction):
        if a == b:
            correct_count += 1
    
    accuracy = correct_count*100 / len(test_labels)
    return accuracy