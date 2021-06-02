# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
This code takes the data from within 
"""
import numpy as np
import knn_utils as kn

train_data = [ [67 , 56, 59],
                  [37 , 48, 51],
                  [77 , 86, 93],
                  [47 , 58, 52],
                  [89 , 77, 69],
                  [40 , 66, 54],
                  [49 , 39, 50],
                  [61 , 37, 43],
                  [78 , 90, 98],
                  [67 , 70, 68]]
    
train_labels=["Average", "JustPass", "Excellent", "JustPass", 
               "Excellent", "Average","JustPass", "JustPass",
               "Excellent","Average"]


test_data = [ [37 , 56, 49],
              [67 , 78, 61],
              [57 , 36, 43],
              [87 , 78, 92],
              [69 , 57, 70]]

test_labels = ["JustPass","Average","JustPass","Excellent", "Average" ]

knn = 1
test_prediction =[]
for test_fv in test_data:
    distance= []
    for train_fv in train_data:
        dist = [abs(a-b) for a,b in zip(test_fv, train_fv)]
        distance.append(sum(dist))
    
    min_val = min(distance)
    min_ind = distance.index(min_val)
    prediction_class = train_labels[min_ind]
    print(test_fv, prediction_class )
    test_prediction.append(prediction_class)
    
print("test prediction classes are :", test_prediction)


correct_count = 0
for a,b in zip(test_labels, test_prediction):
    if a == b:
        correct_count += 1

accuracy = correct_count*100 / len(test_labels)

print ("Test Accuracy (percentage) is:", accuracy)
