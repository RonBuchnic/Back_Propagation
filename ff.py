import numpy as np


def activation(x):
	return np.tanh(x)


def activation_deriv(x):
	return 1-(np.tanh(x)**2)


def loss_deriv(output, target):
	# Derivative of loss function with respect to the activations
	# in the output layer.
	# we use quadratic loss, where L = 0.5*||output-target||^2

	return 2 * (output - target)


class FF(object):
	"""A simple FeedForward neural network"""
	def __init__(self, layerDims):
		super(FF, self).__init__()
		self.n_layer = len(layerDims)
		self.n_weights = len(layerDims)-1
		self.weights = []
		for i in range(self.n_weights):
			self.weights.append(0.1*np.random.randn(layerDims[i+1], layerDims[i]))

	def sgd(self, X, y, epochs, eta, mb_size, Xtest, ytest):
		N = X.shape[1]
		n_mbs = int(np.ceil(N/mb_size))
		acc = self.eval_test(Xtest, ytest)

		updates = 0
		steps = [updates]
		test_acc = [acc]
		print("Starting training, test accuracy: {0}".format(acc))

		for i in range(epochs):
			perm = np.random.permutation(N)
			for j in range(n_mbs):
				X_mb = X[:, perm[j*mb_size:(j+1)*mb_size]]
				y_mb = y[:, perm[j*mb_size:(j+1)*mb_size]]
				grads = self.backprop(X_mb, y_mb)

				for k, grad in enumerate(grads):
					self.weights[k] = self.weights[k] - (eta/mb_size)*grad

				updates = updates + 1
				if updates % 50 == 0:
					steps.append(updates)
					test_acc.append(self.eval_test(Xtest, ytest))

			acc = self.eval_test(Xtest, ytest)
			print("Done epoch {0}, test accuracy: {1}".format(i+1, acc))

		steps = np.asarray(steps)
		steps = steps/n_mbs

		return steps, test_acc

	def predict(self, x):
		a = x
		for w in self.weights:
			a = activation(np.dot(w, a))

		return a

	def eval_test(self, Xtest, ytest):
		ypred = self.predict(Xtest)
		ypred = ypred == np.max(ypred, axis=0)

		return np.mean(np.all(ypred == ytest, axis=0))

	def forward_pass(self, X):
		all_layers = []
		last_layer = X
		all_layers.append(X)
		for w in self.weights:
			s = activation(w @ last_layer)
			all_layers.append(s)
			last_layer = s
		return all_layers

	def backward_pass(self, y, layers):
		all_delta = []
		delta_m = loss_deriv(layers[-1], y) * activation_deriv(self.weights[-1] @ layers[-2])
		all_delta.append(delta_m)
		last_delta = delta_m

		for i in range(self.n_layer-2):
			delta = (self.weights[-1-i].T @ last_delta) * activation_deriv(self.weights[-2-i] @ layers[-3-i])
			all_delta.append(delta)
			last_delta = delta

		return all_delta[::-1]

	def calc_gradients(self, layers, deltas):
		all_gradients = []
		for i in range(self.n_weights):
			all_gradients.append(deltas[i] @ layers[i].T)
		return all_gradients

	def backprop(self, X, y):
		# X is a matrix of size input_dim*mb_size
		# y is a matrix of size output_dim*mb_size
		# you should return a list 'grads' of length(weights) such
		# that grads[i] is a matrix containing the gradients of the
		# loss with respect to weights[i].

		# ForwardPass
		all_layers = self.forward_pass(X)

		# BackwardPass
		all_deltas = self.backward_pass(y, all_layers)

		# Gradients
		return self.calc_gradients(all_layers, all_deltas)


