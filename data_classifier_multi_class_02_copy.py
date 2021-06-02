# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:37:42 2021

@author: IRDC Lab
"""

import matplotlib.pyplot as plt
import numpy as np
import classifier_utils as ut



########## Data Simulation ##############
data_1_array,data_2_array = ut.data_2_class_simulation(
class_1 = (-4,3),class_2 = (3,-6),
class_1_var = 3,class_2_var = 3,
data_1_count = 40, data_2_count = 55)

########## Data Simulation ##############
data_3_array,data_4_array = ut.data_2_class_simulation(
class_1 = (-6,-5),class_2 = (3,5),
class_1_var = 3,class_2_var = 2,
data_1_count = 40, data_2_count = 55)
########## Data Simulation Ends ##############
########## Data Simulation Ends ##############

########## Data Simulation ##############
data_5_array,data_6_array = ut.data_2_class_simulation(
class_1 = (-3,-1),class_2 = (3,0),
class_1_var = 3,class_2_var = 2,
data_1_count = 30, data_2_count = 25)
########## Data Simulation Ends ##############

SHOW_PLOT_DATA = True
if SHOW_PLOT_DATA:
    
    x1_ax =np.arange(-7,8,1)
    #print(x2)
    
    plt.figure(2, figsize=(8, 8))
    
    plt.scatter(data_1_array[0,:],data_1_array[1,:], c="r", marker ='^',
                cmap=plt.cm.Set1, edgecolor='k')
    plt.scatter(data_2_array[0,:],data_2_array[1,:], c="g", marker ='o', 
                cmap=plt.cm.Set1, edgecolor='k')
    
    plt.scatter(data_3_array[0,:],data_3_array[1,:], c="b", marker =',', 
                cmap=plt.cm.Set1, edgecolor='k')
    plt.scatter(data_4_array[0,:],data_4_array[1,:], c="y", marker ='<',
                cmap=plt.cm.Set1, edgecolor='k')
    
    plt.scatter(data_5_array[0,:],data_5_array[1,:], c="k", marker ='.', 
                cmap=plt.cm.Set1, edgecolor='k')
    plt.scatter(data_6_array[0,:],data_6_array[1,:], c="m", marker ='>',
                cmap=plt.cm.Set1, edgecolor='k')
    
    
    
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(["class 1","class 2","class 3", "class 4"])
    plt.title("Multi Classes Data" )
    
    
    #x_min, x_max = min(x1) - 1, max(x1) + 1
    #y_min, y_max = min(x2) - 1, max(x2) + 1
    
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    
    plt.xticks((x1_ax))
    plt.yticks((x1_ax))
    plt.grid()
    plt.show()

################### Train Data Preparation ##########

train_data_label =[]
train_data_label.extend(ut.data_array_2_data_labels(data_1_array,"class_1"))
train_data_label.extend(ut.data_array_2_data_labels(data_2_array,"class_2"))
train_data_label.extend(ut.data_array_2_data_labels(data_3_array,"class_3"))
train_data_label.extend(ut.data_array_2_data_labels(data_4_array,"class_4"))
# train_data_label.extend(ut.data_array_2_data_labels(data_5_array,"class_5"))
# train_data_label.extend(ut.data_array_2_data_labels(data_6_array,"class_6"))



   
    
train_labels= [a[1] for a in train_data_label]
class_names = np.unique(train_labels)

num_classes = len(class_names)

sub_cf=[]
    
OVO_OR_OVA = "ova"
if OVO_OR_OVA == "ovo":

    ################### OVO Training #########################
       
    for cf in range(num_classes):
        for v_cf in range(num_classes):
            if v_cf > cf:
                sub_cf.append([cf, [v_cf] ])
else:
    ################### OVA Training #########################
    for cf in range(num_classes):
        opp_cf =[]
        for v_cf in range(num_classes):
            if v_cf != cf:
                opp_cf.append(v_cf)
                
        sub_cf.append([cf, opp_cf ])
    
                    
print(sub_cf)

sub_cf_trained =[]

for s_cf in sub_cf:    
    data_A = [a[0] for a in train_data_label if a[1] == class_names[s_cf[0]] ]
    data_B=[]
    for b_class in s_cf[1]:    
        data_B.extend([a[0] for a in train_data_label 
                       if a[1] == class_names[b_class] ])

    
    m,c, positive_first = ut.bi_classifier_2d(np.array(data_A), np.array(data_B))
        
    sub_cf_trained.append([s_cf[0],s_cf[1], m, c, positive_first])


    
print (sub_cf_trained)   



########## Data Simulation ##############
data_1_array,data_2_array = ut.data_2_class_simulation(
class_1 = (-4,3),class_2 = (3,-4),
class_1_var = 3,class_2_var = 3,
data_1_count = 5, data_2_count = 5)

########## Data Simulation ##############
data_3_array,data_4_array = ut.data_2_class_simulation(
class_1 = (-4,-5),class_2 = (3,4),
class_1_var = 3,class_2_var = 2,
data_1_count = 5, data_2_count = 5)
########## Data Simulation Ends ##############

################### Test Data Preparation ##########

test_data_label =[]
test_data_label.extend(ut.data_array_2_data_labels(data_1_array,"class_1"))
test_data_label.extend(ut.data_array_2_data_labels(data_2_array,"class_2"))
test_data_label.extend(ut.data_array_2_data_labels(data_3_array,"class_3"))
test_data_label.extend(ut.data_array_2_data_labels(data_4_array,"class_4"))
# test_data_label.extend(ut.data_array_2_data_labels(data_3_array,"class_5"))
# test_data_label.extend(ut.data_array_2_data_labels(data_4_array,"class_6"))


correct =0
total =0
for test_sample in test_data_label:
    test_data = test_sample[0]
    test_label = test_sample[1]
    class_score = np.zeros(num_classes)
    for cf in sub_cf_trained:
        out = ut.bi_classifier_2d_predict([test_data], cf[2], cf[3],cf[4])
        if out[0] == 0:
            class_score[cf[0]]+=1
        else:
            for a in cf[1]:
                class_score[a] += (1/len(cf[1])) 
        print(class_score)
    
    pred_class = class_names[np.argmax(class_score)]
    print(pred_class)
    total +=1
    if pred_class == test_label:
        correct +=1
            
accuracy = correct*100/total

print("accuracy for {} test samples is {} %".format(total,accuracy))            
            
        
        
        
    
    
    






