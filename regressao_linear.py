# -*- coding: utf-8 -*-
import numpy as np

class LinearRegression():
	def __init__(self):
		self.theta_n = []
		self.theta_0 = 0.
		self.loss = []

	#inicializa os pesos aleatoriamente com amostras da distribuição normal
	def init_weights(self, dim):
		return np.random.randn(dim).reshape(dim,1)
		#return np.ones(dim).reshape(dim,1)

	#função de custo
	def loss_function(self, Y, gH, m):
		loss = np.sum(np.power(gH,2))/(2*m)

		return loss

	def prints(self, epoch):
		print "--epoca %s: " % epoch
		print "loss: ", self.loss[epoch]
		print "theta: ", self.theta_0.reshape(theta[0].shape[0]), self.theta_n.reshape(theta[1].shape[0])


	def fit(self, X, Y, epochs=3, learning_rate=0.01, print_results=False):
		#dimensão dos dados
		m = X.shape[0]
		n = X.shape[1]

		#inicializa os pesos aleatoriamente
		self.theta_n = self.init_weights(n)
		self.theta_0 = self.init_weights(1)

		X = X.T
		Y = Y.reshape(1,m)

		#verifica as dimensões
		#assert(self.theta_n.shape[0] == X.shape[0])
		
		for i in xrange(epochs):
			#calcula H
			H = np.dot(self.theta_n.T, X) + self.theta_0

			#calcula gradientes
			gH = H - Y
			
			gTheta_n = np.dot(X, gH.T)/m
			gTheta_0 = np.sum(gH)/m

			#calcula função de custo
			loss = self.loss_function(Y, gH, m)
			self.loss.append(loss)

			#atualiza pesos
			self.theta_0 -= learning_rate*gTheta_0
			self.theta_n -= learning_rate*gTheta_n

			if print_results:
				self.prints(i)

		#calcula função de custo final
		#calcula H
		H = np.dot(self.theta_n.T, X) + self.theta_0

		#calcula gradientes
		gH = H - Y
		loss = self.loss_function(Y, gH, m)
		self.loss.append(loss)

		return self

	def mean_squared_error(self, Y_true, Y_pred):
		return np.power((Y_true - Y_pred),2).mean()

	def predict(self, X):
		X = X.T

		#verifica as dimensões antes de fazer o produto interno
		#assert(self.theta_n.shape[0] == X.shape[0])

		Y_predict = np.dot(self.theta_n.T, X) + self.theta_0
		#Z.shape == (1,m)
		#sigmoid_z.shape = (1,m) -> todas as predições estão neste array

		return Y_predict


