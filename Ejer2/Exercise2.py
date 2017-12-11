import pandas
import numpy as np
import cv2
import math

# General variable
number_k_train = 4
number_k_test = 8

###################
# Training
# K-means right
###################
# Load right data. Iterate over the folder and files with "PasilloDerecha"
# Store all points from all example in one variable
for i in range(0, 9):
    data = pandas.read_csv('Train/PasilloDerecha'+str(i)+'.txt', sep=',', header=None, usecols=[2, 3], engine='python')
    data = data.astype('float32')
    if i == 0:
        ZR = data
    else:
        ZR = np.vstack((ZR, data))


# define criteria and apply kmeans()
criteriaR = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# The main output are the CenterR, there is the number_k_points centers form right data
retR, labelR, centerR = cv2.kmeans(ZR, number_k_train, None, criteriaR, 10, cv2.KMEANS_RANDOM_CENTERS)


######################
# Training
# K-menas left
######################
# Load left data. Iterate over the folder and files with "PasilloIzquierda"
# Store all points from all example in one variable

for i in range(0, 9):
    data = pandas.read_csv('Train/PasilloIzquierda'+str(i)+'.txt', sep=',', header=None, usecols=[2, 3], engine='python')
    data = data.astype('float32')
    if i == 0:
        ZL = data
    else:
        ZL = np.vstack((ZL, data))

# define criteria and apply kmeans()
criteriaL = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retL, labelL, centerL = cv2.kmeans(ZL, number_k_train, None, criteriaL, 10, cv2.KMEANS_RANDOM_CENTERS)


####################
# Test
####################

#Load an example to test
dataT = pandas.read_csv('Test/Paseante4.txt', sep=',', header=None, usecols=[2, 3], engine='python')
dataT = dataT.astype('float32')
dataT = np.vstack((dataT,dataT))


# define criteria and apply kmeans()
criteriaT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retT, labelT, centerT = cv2.kmeans(dataT, number_k_test, None, criteriaT, 10, cv2.KMEANS_RANDOM_CENTERS)

# To decide if somebody is going to right side, move value will get 1 as value or as nearest as possible. If it's to the
# left, move variable should value 0 or as nearest as possible.

# This variable will represent the direccion
move = 0

# Iterate over the number_k_test points detected.
for i in range(0,centerT.shape[0]):
    # Reset the nearest distance between points in test and train right
    nearestR = 99
    # Iterate over number_k_train on the right side
    for t in range(0,centerR.shape[0]):
        # Calculate distance between points, test and train
        distR = math.sqrt((centerT[i,0]-centerR[t,0]) ** 2 + (centerT[i,1]-centerR[t,1]) ** 2)
        # Get the nearest distance
        if distR<nearestR:
            nearestR=distR

    # Reset the nearest distance between points in test and train left
    nearestL = 99
    # Iterate over number_k_train on the left side
    for m in range(0, centerR.shape[0]):
        # Calculate distance between points, test and train
        distL = math.sqrt((centerT[i, 0] - centerL[m, 0]) ** 2 + (centerT[i, 1] - centerL[m, 1]) ** 2)
        # Get the nearest distance
        if distL < nearestL:
            nearestL = distL
    # Print the nearest distances on right and left side
    # print('R=', nearestR,'L=', nearestL)

    # If test point is nearer to right side than left side, add probability to right side
    if nearestL > nearestR:
        move = move+(1/centerT.shape[0])

    # Next two ifs are implemented to correct if right side and left side are similar. In this case the probability
    # added is just the middle so, it stays it in the middle
    # If it is nearer to right but not too much, remove some probability
    if nearestR*1.2 > nearestL > nearestR:
        move = move - (1/(2*centerT.shape[0]))

    # If test point is nearer to left but not too much, add some probability
    if nearestL*1.2 > nearestR > nearestL:
        move = move + (1 / (2 * centerT.shape[0]))

# Once each number_k_test points from sample are calculated, the prediction is done
# Because get an one or a zero is too strict, there are set a margin of 25%
# This line prints the move's probability
# print(move)
if move >= 0.75:
    print('To the right side')
elif move <= 0.25:
    print('To the left side')
else:
    print('WARNING, CALL POLICE. SOMEBODY IS HANGING AROUND')
