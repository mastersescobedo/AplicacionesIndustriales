from hmmlearn import hmm
import numpy as np

np.random.seed(42)

model = hmm.MultinomialHMM(3,verbose=True,n_iter=20)
states=["Sol","Nublado","Lluvia"]
observaciones=["paraguas","sin paraguas"]

model.start_probability = np.array([0.6, 0.3, 0.1]) #Numeros aleatorios
model.transition_probability = np.array([[0.5, 0.4, 0.1],  [0.5, 0.3,0.2] ,[0.2,0.4,0.4]]) #Poner numeros aleatorios
model.emissionprob = np.array([[0.1, 0.9],  [0.5, 0.5],  [0.8, 0.2]]) #Numeros aleatorios

train1 = [1,1,1,0]
train2 = [0,0,0]
print("train1 =",",".join(map(lambda x: observaciones[x],train1)))
print("train2 =",",".join(map(lambda x: observaciones[x],train2)))
#train1 = 'sin paraguas','sin paraguas','sin paraguas','paraguas'
#train2 = 'paraguas','paraguas','paraguas'

# a los centroides asignar letras
#Coger los numeros de cada paso del train, comparar con que letra está mas cerca para generar secuencia de letras,
# los pongo todos en una tira diciendo donde acaba cada uno.

X = [train1,train2]
lengths = list(map(lambda x : len(x), X))
X = np.hstack(X)
X = X.reshape(len(X),1)
model.fit(X,lengths)

print(model.monitor_)
# print("Se logoro convergencia: ", model.monitor_.converged)
# #ConvergenceMonitor(history=[-3.3234444289327238, -3.314647236558061], iter=15, n_iter=20, tol=0.01, verbose=True)
#
train1_reformated = list(map(lambda x:[x],train1))
train2_reformated = list(map(lambda x:[x],train1))

print("log(P(train1))=",model.score(train1_reformated))
print("log(P(train2))=",model.score(train2_reformated))
#
test = [[0],[0],[1],[1]]
print("log(P(test))=",model.score(test))
# #log(P(train1))= -2.61210562279
# #log(P(train2))= -0.697197857927
# #log(P(test))= -10.0318971295

prediction = model.predict(train1_reformated)

test = [[1],[0],[0],[0],[0],[0],[0],[1]]

prediction = model.predict(test)

print("Prediccion(test) =",",".join(map(lambda x: states[x],prediction)))
prediction = model.predict(train2_reformated)
# print("Prediccion(train2) =",",".join(map(lambda x: states[x],prediction)))
prediction = model.predict(test)
# print("Prediccion(test) =",",".join(map(lambda x: states[x],prediction)))
#
# print(model.startprob_)
# print(model.transmat_)
# print(model.emissionprob_)