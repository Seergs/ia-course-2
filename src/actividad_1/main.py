import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, n_input, learning_rate):
        self.w = -1 + 2 * np.random.rand(n_input)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate

    def predict(self, X):
        p = X.shape[1]
        y_est = np.zeros(p)
        for i in range(p):
            y_est[i] = np.dot(self.w, X[:,i]) + self.b
            if y_est[i] >= 0:
                y_est[i] = 1
            else:
                y_est[i] = 0
        
        return y_est

    def train(self, X, Y, epochs=50):
        p = X.shape[1]
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:, i].reshape(-1, 1))
                self.w += self.eta * (Y[i] - y_est) * X[:, i]
                self.b += self.eta * (Y[i] - y_est)


def draw_2d(model):
    w1, w2, b  = model.w[0], model.w[1], model.b
    li, ls = -2, 2
    plt.plot([li, ls], 
        [(1 / w2) * (-w1 * li - b), 
        (1 / w2) * (-w1 * ls - b)], 
        '--k')
    plt.show()

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

Y = np.array([0, 0, 0, 1])

neuron = Perceptron(n_input=2, learning_rate=0.1)

neuron.train(X, Y)

_, p = X.shape

for i in range(p):
    if Y[i] == 0:
        plt.plot(X[0, i], X[1, i], 'or')
    else:
        plt.plot(X[0, i], X[1, i], 'ob')

plt.title("Perceptrón")
plt.grid('on')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel(r'$x_1$')
plt.xlabel(r'$x_2$')

draw_2d(neuron)