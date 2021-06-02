# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:37:42 2021

@author: IRDC Lab
"""
import matplotlib.pyplot as plt
import numpy as np

### Equation  x2 = mx1 + c  where m is slope and c is y intercept
m = -2
c = 1

x1 =np.arange(-7,8,1)
x2 = [m*x + c for x in x1]

plt.figure(2, figsize=(8, 6))
plt.scatter(0,0, c="b",   cmap=plt.cm.Set1, edgecolor='k')
plt.plot(x1,np.zeros(len(x1)),'.-b')
plt.plot(np.zeros(len(x1)),x1,'.-b')
plt.plot(x1,x2,'.-r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(["x-axis", "y-axis", "line"])
plt.title("Line y=mx+c ; m="+str(m)+" & c= " + str(c) )

plt.xlim(-7, 7)
plt.ylim(-7, 7)

plt.xticks((x1))
plt.yticks((x1))
plt.grid(True)
plt.show()



