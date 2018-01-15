import pandas
import numpy as np
import cv2
from hmmlearn import hmm
import math


np.random.seed(42)

# General variable
number_k_train = 3
number_k_test = 3





######################
# Training
# K-menas left
######################
# Load left data. Iterate over the folder and files with "PasilloIzquierda"
# Store all points from all example in one variable

ZL_list = [None]*9

for i in range(0, 9):
    data = np.loadtxt('Train/PasilloDerecha' + str(i) + '.txt', delimiter=',', usecols=[2, 3])
    data = data.astype('float32')
    if i == 0:
        ZL_np = data
    else:
        ZL_np = np.vstack((ZL_np, data))
    data_list = data.tolist()
    ZL_list[i] = data_list

# define criteria and apply kmeans()
criteriaL = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retL, labelL, centerL = cv2.kmeans(ZL_np, number_k_train, None, criteriaL, 10, cv2.KMEANS_RANDOM_CENTERS)

full_secuenceL = [None]*9
secuenceL = list()
lengthsL = list()

for s in range(0, 9):
    data = np.loadtxt('Train/PasilloDerecha'+str(s)+'.txt', delimiter=',', usecols=[2, 3])
    data = data.astype('float32')
    ZL = np.asarray(data)
    lengthsL.append(ZL.shape[0])

    secuenceL = list()
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
    full_secuenceL[s]=secuenceL


#Left model

modelL = hmm.MultinomialHMM(2,verbose=True,n_iter=20)

modelL.start_probability = np.array([0.5, 0.5]) #Numeros aleatorios
modelL.transition_probability = np.array([[0.3, 0.7],  [0.7, 0.3]]) #Poner numeros aleatorios
modelL.emissionprob = np.array([[0.7, 0.3],  [0.3, 0.7]]) #Numeros aleatorios


# modelL = hmm.MultinomialHMM(2,verbose=True,n_iter=20)
#
# modelL.start_probability = np.array([0.6, 0.3, 0.1]) #Numeros aleatorios
# modelL.transition_probability = np.array([[0.5, 0.4, 0.1],  [0.5, 0.3,0.2] ,[0.2,0.4,0.4]]) #Poner numeros aleatorios
# modelL.emissionprob = np.array([[0.1, 0.9],  [0.5, 0.5],  [0.8, 0.2]]) #Numeros aleatorios

X = np.asarray(full_secuenceL)

#lengths = list(map(lambda x : len(x), X))
X = np.hstack(X)
X = X.reshape(len(X),1)
modelL.fit(X,lengthsL)



####################
# Test
####################

secuenceTR = list()
secuenceTL = list()
lengthsT = list()

#Load an example to test
dataT = pandas.read_csv('Test/PasilloDerecha12.txt', sep=',', header=None, usecols=[2, 3], engine='python')
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

    # for t in range(0, centerR.__len__()):
    #     # Calculate distance between points, test and train
    #     distR = math.sqrt((dataT[i, 0] - centerR[t, 0]) ** 2 + (dataT[i, 1] - centerR[t, 1]) ** 2)
    #     # Get the nearest distance
    #     if distR < nearestR:
    #         nearestR = distR
    #         aux = t
    # secuenceTR.append(aux)

secuenceTest = [None]*1
secuenceTest[0]=secuenceTL

TL = np.asarray(secuenceTest)
#lengths = list(map(lambda x : len(x), X))
TL = np.hstack(TL)
TL = TL.reshape(len(TL),1)

predictionL = modelL.predict(TL)

# PArece que el cero es izquierda
# Parece que 2 es derecha

unique, counts = np.unique(predictionL, return_counts=True)
resumen = dict(zip(unique, counts))


'''


modelo 1
train izquierda + derecha = suma=0
train izquierda + izquierda = suma>0
train izquierda + paseante = suma>0

train derecha + derecha = suma>0 todos menos uno
train derecha + izquierda = suma=0
train derecha + paseante = suma>0



'''

print(predictionL)
suma =np.sum(predictionL)
print(suma)

