import pandas
import numpy as np
import cv2
from hmmlearn import hmm
import math


np.random.seed(42)

# General variable
number_k_train = 3
number_k_test = 3

###################
# Training
# K-means right
###################
# Load right data. Iterate over the folder and files with "PasilloDerecha"
# Store all points from all example in one variable
trainR = []

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

secuenceR = list()
lengthsR = list()

for i in range(0, 9):
    data = pandas.read_csv('Train/PasilloDerecha'+str(i)+'.txt', sep=',', header=None, usecols=[2, 3], engine='python')
    data = data.astype('float32')
    ZR = np.asarray(data)
    lengthsR.append(ZR.shape[0])

    # Iterate over the number_k_test points detected.
    for i in range(0, ZR.shape[0]):
        # Reset the nearest distance between points in test and train right
        nearestR = 99
        # Iterate over number_k_train on the right side
        for t in range(0, centerR.__len__()):
            # Calculate distance between points, test and train
            distR = math.sqrt((ZR[i, 0] - centerR[t, 0]) ** 2 + (ZR[i, 1] - centerR[t, 1]) ** 2)
            # Get the nearest distance
            if distR < nearestR:
                nearestR = distR
                aux=t
        secuenceR.append(aux)

######################
# Training
# K-menas left
######################
# Load left data. Iterate over the folder and files with "PasilloIzquierda"
# Store all points from all example in one variable

for i in range(0, 9):
    data = pandas.read_csv('Train/PasilloIzquierda' + str(i) + '.txt', sep=',', header=None, usecols=[2, 3],
                           engine='python')
    data = data.astype('float32')
    if i == 0:
        ZL = data
    else:
        ZL = np.vstack((ZL, data))

# define criteria and apply kmeans()
criteriaL = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retL, labelL, centerL = cv2.kmeans(ZL, number_k_train, None, criteriaL, 10, cv2.KMEANS_RANDOM_CENTERS)


secuenceL = list()
lengthsL = list()

for i in range(0, 9):
    data = pandas.read_csv('Train/PasilloDerecha'+str(i)+'.txt', sep=',', header=None, usecols=[2, 3], engine='python')
    data = data.astype('float32')
    ZL = np.asarray(data)
    lengthsL.append(ZL.shape[0])

    # Iterate over the number_k_test points detected.
    for i in range(0, ZL.shape[0]):
        # Reset the nearest distance between points in test and train right
        nearestL = 99
        # Iterate over number_k_train on the right side
        for t in range(0, centerL.__len__()):
            # Calculate distance between points, test and train
            distL = math.sqrt((ZL[i, 0] - centerL[t, 0]) ** 2 + (ZL[i, 1] - centerL[t, 1]) ** 2)
            # Get the nearest distance
            if distL < nearestL:
                nearestL = distL
                aux=t
        secuenceL.append(aux)





# Right model

modelR = hmm.MultinomialHMM(2,verbose=True,n_iter=20)

modelR.start_probability = np.array([0.6, 0.4]) #Numeros aleatorios
modelR.transition_probability = np.array([[0.5, 0.5],  [0.5, 0.5]]) #Poner numeros aleatorios
modelR.emissionprob = np.array([[0.1, 0.5, 0.4],  [0.5, 0.3, 0.2]]) #Numeros aleatorios

X = np.asarray(secuenceR)
#lengths = list(map(lambda x : len(x), X))
X = np.hstack(X)
X = X.reshape(len(X),1)
modelR.fit(X,lengthsR)

print('a')


# Left model

modelL = hmm.MultinomialHMM(3,verbose=True,n_iter=20)

modelL.start_probability = np.array([0.6, 0.3, 0.1]) #Numeros aleatorios
modelL.transition_probability = np.array([[0.5, 0.4, 0.1],  [0.5, 0.3,0.2] ,[0.2,0.4,0.4]]) #Poner numeros aleatorios
modelL.emissionprob = np.array([[0.1, 0.9],  [0.5, 0.5],  [0.8, 0.2]]) #Numeros aleatorios

P = np.asarray(secuenceL)
#lengths = list(map(lambda x : len(x), X))
P = np.hstack(P)
P = P.reshape(len(P),1)
modelL.fit(P,lengthsL)

print('b')




####################
# Test
####################

secuenceTR = list()
secuenceTL = list()
lengthsT = list()

#Load an example to test
dataT = pandas.read_csv('Test/Paseante4.txt', sep=',', header=None, usecols=[2, 3], engine='python')
dataT = dataT.astype('float32')
dataT = np.asarray(dataT)
lengthsT.append(dataT.shape[0])



#######################
# Data to centroids
#######################

# Iterate over the number_k_test points detected.
for i in range(0, dataT.shape[0]):
    # Reset the nearest distance between points in test and train right
    nearestL = 99
    nearestR = 99
    # Iterate over number_k_train on the right side
    for t in range(0, centerL.__len__()):
        # Calculate distance between points, test and train
        distL = math.sqrt((dataT[i, 0] - centerL[t, 0]) ** 2 + (dataT[i, 1] - centerL[t, 1]) ** 2)
        # Get the nearest distance
        if distL < nearestL:
            nearestL = distL
            aux=t
    secuenceTL.append(aux)

    for t in range(0, centerR.__len__()):
        # Calculate distance between points, test and train
        distR = math.sqrt((dataT[i, 0] - centerR[t, 0]) ** 2 + (dataT[i, 1] - centerR[t, 1]) ** 2)
        # Get the nearest distance
        if distR < nearestR:
            nearestR = distR
            aux = t
    secuenceTR.append(aux)


TL = np.asarray(secuenceTL)
#lengths = list(map(lambda x : len(x), X))
TL = np.hstack(TL)
TL = TL.reshape(len(TL),1)

TR = np.asarray(secuenceTR)
#lengths = list(map(lambda x : len(x), X))
TR = np.hstack(TR)
TR = TR.reshape(len(TR),1)

predictionR = modelR.predict(secuenceTR)

predictionL = modelL.predict(secuenceTL)


























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


