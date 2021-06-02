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
class_1 = (-4,3),class_2 = (3,-4),
class_1_var = 3,class_2_var = 3,
data_1_count = 40, data_2_count = 55)

########## Data Simulation ##############
data_3_array,data_4_array = ut.data_2_class_simulation(
class_1 = (-4,-5),class_2 = (3,4),
class_1_var = 3,class_2_var = 2,
data_1_count = 40, data_2_count = 55)
########## Data Simulation Ends ##############
########## Data Simulation Ends ##############


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

#print("hyper-line Equation y=mx +c, where m is {} & c is {}".format(slope_of_hyper_line,ch))



