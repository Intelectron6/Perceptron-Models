#Single Layered Perceptron Classifier

import numpy as np
import matplotlib.pyplot as plt

def predict(X, w, b):
    return np.sign(X.dot(w) + b)

def score(X, Y, w, b):
    P = predict(X, w, b)
    return np.mean(P == Y)

def fit(X, Y, learning_rate, epoch):
    D = X.shape[1]
    w = np.random.randn(D)
    b = 0
    N = len(Y)
    costs = []

    for e in range(epoch):
        Yp = predict(X, w, b)
        incorrect = np.nonzero(Y != Yp)[0]
        if len(incorrect) == 0:
            break
        i = np.random.choice(incorrect)
        w = w + learning_rate*Y[i]*X[i]
        b = b + learning_rate*Y[i]

        c = len(incorrect)/float(N)
        costs.append(c)
        
    print(e+1)
    plt.plot(costs)
    plt.show()
    return w,b

w1 = np.array([-0.5, 0.5])
b1 = 0.1
#X = np.random.random((1000,2))*2
#Y = np.sign(X.dot(w1) + b1)

N = 1000
D = 2
X = np.random.randn(N,D)
X[:500,:] = X[:500,:] - 1*np.ones((500,D))
X[500:,:] = X[500:,:] + 1*np.ones((500,D))
Y = np.array([0]*500 + [1]*500)

plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()

learning_rate = 1
epoch = 1000

X_train, Y_train = X[:750], Y[:750]
X_test, Y_test = X[750:], Y[750:]

w,b = fit(X_train, Y_train, learning_rate, epoch)
#print('Accuracy on training set =',score(X_train, Y_train, w, b)*100)
print('Accuracy on testing set =',score(X_test, Y_test, w, b)*100)

