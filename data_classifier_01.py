# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:37:42 2021

@author: IRDC Lab
"""

import matplotlib.pyplot as plt
import numpy as np
import classifier_utils as ut


################ ML Algorithm ##############################
# Hyper-line (yellow) is  perpendicular to  centroid line(blue)
# So, Slope of  hyper-line = -1/ slope of centroid line
# We know two points of centroid line, its slope =   (4-2)/ (4- (-1)) = 2/5
# So, slope of hyper-line = -5/2 
# We know standard point(x1,y1)-slope(m) line equation 
# i.e. y-y1 = m(x-x1)    where (x1,y1) = (3/2,3) & m = 5/2 
# So,   y-3 = -5/2(x -  3/2)     4y – 12= -10x + 15 
# So,   4y+10x -27 = 0 is the hyper-line 
################ ML Algorithm Ends ##############################


########## Data Simulation ##############
data_1_array,data_2_array = ut.data_2_class_simulation(
class_1 = (-4,-1),class_2 = (3,-4),
class_1_var = 4,class_2_var = 3,
data_1_count = 40, data_2_count = 55)
########## Data Simulation Ends ##############

centroid_1 = np.mean(data_1_array, axis = 1)
centroid_2 = np.mean(data_2_array, axis = 1)

centroids_mid_point = np.add(centroid_1,centroid_2)/2

slope_of_centroid_line = (centroid_2[1]- centroid_1[1])/(centroid_2[0]- centroid_1[0])

slope_of_hyper_line = -1/slope_of_centroid_line
xc,yc = ut.line_draw(slope_of_centroid_line, centroid_1[1] - (slope_of_centroid_line*centroid_1[0]))

## y-y1 = m(x-x1)
## y = mx - mx1 + y1 =>  y = mx + c where c = (y1 - mx1)
x1, y1 = centroids_mid_point[0],centroids_mid_point[1]


ch = centroids_mid_point[1]- (slope_of_hyper_line*centroids_mid_point[0])

xh,yh = ut.line_draw(slope_of_hyper_line,ch)


x1_ax =np.arange(-7,8,1)
#print(x2)

plt.figure(2, figsize=(8, 8))

plt.scatter(data_1_array[0,:],data_1_array[1,:], c="r", marker ='^',  cmap=plt.cm.Set1, edgecolor='k')
plt.scatter(data_2_array[0,:],data_2_array[1,:], c="g", marker ='o',  cmap=plt.cm.Set1, edgecolor='k')

plt.plot(centroid_1[0],centroid_1[1],'v-g')
plt.plot(centroid_2[0],centroid_2[1],'v-r')

plt.plot(x1_ax,np.zeros(len(x1_ax)),'.-b')
plt.plot(np.zeros(len(x1_ax)),x1_ax,'.-b')
plt.plot(x1,y1,'o-k')
plt.plot(xc,yc,'.-g')
plt.plot(xh,yh,'.-r')
plt.xlabel('X')
plt.ylabel('Y')
#plt.legend(["x-axis", "y-axis", "line"])
plt.title("Data Classification using Simple ML" )


#x_min, x_max = min(x1) - 1, max(x1) + 1
#y_min, y_max = min(x2) - 1, max(x2) + 1

plt.xlim(-7, 7)
plt.ylim(-7, 7)

plt.xticks((x1_ax))
plt.yticks((x1_ax))
plt.grid()
plt.show()

print("hyper-line Equation y=mx +c, where m is {} & c is {}".format(slope_of_hyper_line,ch))



