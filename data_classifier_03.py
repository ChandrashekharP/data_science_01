# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:37:42 2021

@author: IRDC Lab
"""

import matplotlib.pyplot as plt
import numpy as np
import classifier_utils as ut
from mpl_toolkits.mplot3d import Axes3D


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
data_1_array,data_2_array = ut.data3D_2_class_simulation(
class_1 = (-2,1,3),class_2 = (5,4,-1),
class_1_var = 4,class_2_var = 3,
data_1_count = 40, data_2_count = 55)
########## Data Simulation Ends ##############

centroid_1 = np.mean(data_1_array, axis = 1)
centroid_2 = np.mean(data_2_array, axis = 1)


normal_vector = np.subtract(centroid_1,centroid_2)

#xyz_points = np.array(ut.plane_draw(normal_vector))



fig = plt.figure(1, figsize=(8, 8))
ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(data_1_array[0, :], data_1_array[1, :],data_1_array[2, :],
           c="r", marker ='^', cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.scatter(data_2_array[0, :], data_2_array[1, :],data_2_array[2, :],
           c="g", marker ='o', cmap=plt.cm.Set1, edgecolor='k', s=40)

#ax.scatter(xyz_points[:,0], xyz_points[:,1],xyz_points[:,2],
   #        c="y", marker ='o', cmap=plt.cm.Set1, edgecolor='k', s=40)

#ax.plot(xyz_points[:,0], xyz_points[:,1],xyz_points[:,2],'o-y')

ax.plot(centroid_1[0],centroid_1[1],centroid_1[2],'^-g')
ax.plot(centroid_2[0],centroid_2[1],centroid_2[2],'o-r')

ax.plot([centroid_1[0], centroid_2[0]],
        [centroid_1[1], centroid_2[1]],
        [centroid_1[2], centroid_2[2]],
         '-b')


ax.set_title("3D Data Classifier")
ax.set_xlabel("X")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Y")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Z")
ax.w_zaxis.set_ticklabels([])

plt.show()

