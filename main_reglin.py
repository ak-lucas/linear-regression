# -*- coding: utf-8 -*-

import csv
import numpy as np
from regressao_linear import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#CARREGA DATASET
diabetes = datasets.load_diabetes()
nb_instances = diabetes.data.shape[0]

#EMBARALHA
indices = np.arange(nb_instances)

np.random.shuffle(indices)

X = diabetes.data[indices]
Y = diabetes.target[indices]

#SEPARA CONJUNTOS DE TREINO E TESTE
TRAIN_SIZE = int(.8 * nb_instances)

X_train = X[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]

X_test = X[TRAIN_SIZE:]
Y_test = Y[TRAIN_SIZE:]

lr = LinearRegression()

lr.fit(X_train, Y_train, epochs=10000, learning_rate=0.1)

Y_predict = lr.predict(X_test)

print "loss function: ", lr.loss[-1]
print "MSE: ", lr.mean_squared_error(Y_predict, Y_test)
#print mean_squared_error(Y_test.T, Y_predict.T)	#sklearn