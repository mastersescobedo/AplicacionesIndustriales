import pandas
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

###################
# K-menas right
###################
for i in range(0,9):
    data = pandas.read_csv('Train/PasilloDerecha'+str(i)+'.txt', sep=',', header=None, usecols=[2,3], engine='python')
    data = data.astype('float32')
    if i==0:
        ZR = data
    else:
        ZR = np.vstack((ZR,data))
#print(data)

# define criteria and apply kmeans()
criteriaR = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retR,labelR,centerR=cv2.kmeans(ZR,4,None,criteriaR,10,cv2.KMEANS_RANDOM_CENTERS)
# Now separate the data, Note the flatten()
AR = ZR[labelR.ravel()==0]
BR = ZR[labelR.ravel()==1]
CR = ZR[labelR.ravel()==2]
DR = ZR[labelR.ravel()==3]

# # Plot the data
# plt.scatter(AR[:,0],AR[:,1])
# plt.scatter(BR[:,0],BR[:,1],c = 'r')
# plt.scatter(CR[:,0],CR[:,1],c = 'g')
# plt.scatter(DR[:,0],DR[:,1],c = 'y')
#
# plt.scatter(centerR[:,0],centerR[:,1],s = 80,c = 'b', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()

######################
# K-menas left
######################
for i in range(0,9):
    data = pandas.read_csv('Train/PasilloIzquierda'+str(i)+'.txt', sep=',', header=None, usecols=[2,3], engine='python')
    data = data.astype('float32')
    if i==0:
        ZL = data
    else:
        ZL = np.vstack((ZL,data))
#print(data)

# define criteria and apply kmeans()
criteriaL = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retL,labelL,centerL=cv2.kmeans(ZL,4,None,criteriaL,10,cv2.KMEANS_RANDOM_CENTERS)
# Now separate the data, Note the flatten()
AL = ZL[labelL.ravel()==0]
BL = ZL[labelL.ravel()==1]
CL = ZL[labelL.ravel()==2]
DL = ZL[labelL.ravel()==3]
#
# # Plot the data
# plt.scatter(AL[:,0],AL[:,1])
# plt.scatter(BL[:,0],BL[:,1],c = 'r')
# plt.scatter(CL[:,0],CL[:,1],c = 'g')
# plt.scatter(DL[:,0],DL[:,1],c = 'y')
#
# plt.scatter(centerL[:,0],centerL[:,1],s = 80,c = 'b', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()


####################
#Walk
####################

dataW = pandas.read_csv('Test/Paseante4.txt', sep=',', header=None, usecols=[2,3], engine='python')
dataW = dataW.astype('float32')
dataW = np.vstack((dataW,dataW))


# define criteria and apply kmeans()
criteriaW = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retW,labelW,centerW=cv2.kmeans(dataW,8,None,criteriaW,10,cv2.KMEANS_RANDOM_CENTERS)
# Now separate the data, Note the flatten()
# AW = ZR[labelW.ravel()==0]
# BW = ZR[labelW.ravel()==1]
# CW = ZR[labelW.ravel()==2]
# DW = ZR[labelW.ravel()==3]

# # Plot the data
# plt.scatter(AR[:,0],AR[:,1])
# plt.scatter(BR[:,0],BR[:,1],c = 'r')
# plt.scatter(CR[:,0],CR[:,1],c = 'g')
# #plt.scatter(D[:,0],D[:,1],c = 'y')
#
# plt.scatter(centerW[:,0],centerW[:,1],s = 80,c = 'b', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()

# RIGHT = 0
# LEFT = 1
move = 0

for i in range(0,centerW.shape[0]):
    nearestR = 99
    for t in range(0,centerR.shape[0]):
        distR = math.sqrt((centerW[i,0]-centerR[t,0]) ** 2 + (centerW[i,1]-centerR[t,1]) ** 2)
        if distR<nearestR:
            nearestR=distR

    nearestL = 99
    for m in range(0, centerR.shape[0]):
        distL = math.sqrt((centerW[i, 0] - centerL[m, 0]) ** 2 + (centerW[i, 1] - centerL[m, 1]) ** 2)
        if distL < nearestL:
            nearestL = distL

    print('R=',nearestR,'L=',nearestL)
    if nearestL>nearestR:
        move = move+(1/centerW.shape[0])
        #Sin lo de debajo, solo falla en izquierda12 y derecha 11 limites en 0.8 y 0.2 respectivamente
    if (nearestL<(nearestR*1.2) and nearestL>nearestR):
        move= move - (1/(2*centerW.shape[0]))
    if (nearestR<(nearestL*1.2) and nearestR>nearestL):
        move = move + (1 / (2 * centerW.shape[0]))

if move>=0.75:
    print('To the right side',move)
elif move<=0.25:
    print('To the left side',move)
else:
    print('WARNING, CALL POLICE',move)
