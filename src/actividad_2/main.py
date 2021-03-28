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

elem = 100

X = np.array(([np.random.uniform(1.40, 2.1) for i in range(elem)],
              [np.random.randint(35, 120) for i in range(elem)]))

imc = np.array([X[1, i] / X[0, i]**2 for i in range(elem)])

temp = []
for i in range(elem):
    if imc[i] > 25:
        temp.append(1)
    else:
        temp.append(0)

Y = np.array(temp)

altura_maxima = X[0].min()
altura_minima = X[0].max()
masa_maxima = X[1].min()
masa_minima = X[1].max()

X = np.array(([((X[0, i] - altura_minima) / (altura_maxima - altura_minima)) for i in range(elem)],
            [((X[1, i] - masa_minima) / (masa_maxima - masa_minima)) for i in range(elem)]))

neuron = Perceptron(n_input=2, learning_rate=0.1)

neuron.train(X, Y)

_, p = X.shape
for i in range(p):
    if Y[i] == 0:
        plt.plot(X[0, i], X[1, i], 'or')
    else:
        plt.plot(X[0, i], X[1, i], 'ob')
    

plt.title("Perceptrón IMC")
plt.grid('on')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$x_1$')
plt.xlabel(r'$x_2$')

draw_2d(neuron)