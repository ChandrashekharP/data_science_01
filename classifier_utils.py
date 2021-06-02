# -*- coding: utf-8 -*-
"""
Created on Thu May 27 19:12:07 2021

@author: IRDC Lab
"""
import numpy as np
import itertools

def data_2_class_simulation(class_1 = (-4,-1),class_2 = (2,3),
                            class_1_var = 2,class_2_var = 3,
                            data_1_count = 20, data_2_count = 25):
    
    class_1_d = [class_1[0]*np.ones(data_1_count),
                 class_1[1]*np.ones(data_1_count)]
    class_2_d = [class_2[0]*np.ones(data_2_count),
                 class_2[1]*np.ones(data_2_count)]
    
    class_1_noise =  [class_1_var*np.random.rand(data_1_count),
                      class_1_var*np.random.rand(data_1_count)]
    class_2_noise =  [class_2_var*np.random.rand(data_2_count),
                      class_2_var*np.random.rand(data_2_count)]
    data_1_array =  np.add(np.array(class_1_d), np.array(class_1_noise))
    data_2_array =  np.add(np.array(class_2_d), np.array(class_2_noise))
    
    return data_1_array, data_2_array

def line_draw ( m, c):
    ### Equation  x2 = mx1 + c  where m is slope and c is y intercept
    
    #m = -2
    #c = -2
    
    x1 =np.arange(-7,8,1)
    
    x2 = [m*x + c for x in x1]
    
    return x1,x2

    
def data3D_2_class_simulation(class_1 = (-4,-1, 3),class_2 = (2,3,-1),
                            class_1_var = 2,class_2_var = 3,
                            data_1_count = 20, data_2_count = 25):
    
    class_1_d = [class_1[0]*np.ones(data_1_count),
                 class_1[1]*np.ones(data_1_count),
                 class_1[1]*np.ones(data_1_count)]
    class_2_d = [class_2[0]*np.ones(data_2_count),
                 class_2[1]*np.ones(data_2_count),
                 class_2[1]*np.ones(data_2_count)]
    
    class_1_noise =  [class_1_var*np.random.rand(data_1_count),
                      class_1_var*np.random.rand(data_1_count),
                      class_1_var*np.random.rand(data_1_count)]
    class_2_noise =  [class_2_var*np.random.rand(data_2_count),
                      class_2_var*np.random.rand(data_2_count),
                      class_2_var*np.random.rand(data_2_count)]
    data_1_array =  np.add(np.array(class_1_d), np.array(class_1_noise))
    data_2_array =  np.add(np.array(class_2_d), np.array(class_2_noise))
    
    return data_1_array, data_2_array

def plane_draw (norm_vec ):
    ### Equation  z = ax +by + c  
    
    a=norm_vec[0]
    b=norm_vec[1]
    c=norm_vec[2]
    
    x1 =np.arange(-7,8,1)
    x2 =np.arange(-7,8,1)   
    
    
    points_3d_array = [np.array([x,y,(a*x + b*y+ c)]) for x,y in list(itertools.product(x1,x2))]
    
    return points_3d_array


def bi_classifier_2d(data_1_array, data_2_array):
    
# from  data_classifier_02.py    

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
    # data_1_array,data_2_array = data_2_class_simulation(
    # class_1 = (-4,-1),class_2 = (3,-4),
    # class_1_var = 4,class_2_var = 3,
    # data_1_count = 40, data_2_count = 55)
    ########## Data Simulation Ends ##############
    
    
    centroid_1 = np.mean(data_1_array, axis = 0)
    centroid_2 = np.mean(data_2_array, axis = 0)
    
    centroids_mid_point = np.add(centroid_1,centroid_2)/2
    
    slope_of_centroid_line = (centroid_2[1]- centroid_1[1])/(centroid_2[0]- centroid_1[0])
    
    slope_of_hyper_line = -1/slope_of_centroid_line
    #xc,yc = ut.line_draw(slope_of_centroid_line, centroid_1[1] - (slope_of_centroid_line*centroid_1[0]))
    
    ## y-y1 = m(x-x1)
    ## y = mx - mx1 + y1 =>  y = mx + c where c = (y1 - mx1)
    #x1, y1 = centroids_mid_point[0],centroids_mid_point[1]
    
    
    ch = centroids_mid_point[1]- (slope_of_hyper_line*centroids_mid_point[0])
    
    val_1 = np.mean(bi_classifier_2d_val(data_1_array, slope_of_hyper_line, ch))
    #val_2 = np.mean(bi_classifier_2d_predict(data_2_array, slope_of_hyper_line, ch)) 
    
    #xh,yh = line_draw(slope_of_hyper_line,ch)
    
    return slope_of_hyper_line, ch, np.sign(val_1)


def data_array_2_data_labels(data_array,label):
    train_data = []
    [train_data.append(([x,y],label))
     for x,y in zip(data_array[0],data_array[1])]
    return train_data


def bi_classifier_2d_val(data_array, m, c):
    out=[]
    for data_pt in data_array:
         #  y - mx -c = 0 
         out.append(data_pt[1] - m *data_pt[0] - c)
    return out

def bi_classifier_2d_predict(data_array, m, c, positive_first):
    out=[]
    for data_pt in data_array:
         #  y - mx -c = 0
         val = data_pt[1] - m *data_pt[0] - c
         if val >= 0:             
             if positive_first >0: 
                 out.append(0)
             else:
                 out.append(1)
         else:
            if positive_first > 0: 
                 out.append(1)
            else:
                 out.append(0)
    return out
         