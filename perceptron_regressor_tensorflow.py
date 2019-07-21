#Single Layered Perceptron Regressor using Tensorflow

import sys

if not sys.warnoptions:
    import warnings, os
    warnings.simplefilter("ignore")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0,10,100)
Y = np.linspace(10,50,100) + 3*np.random.randn(100)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

X_train = np.reshape(X_train, [75,1])
X_test = np.reshape(X_test, [25,1])
Y_train = np.reshape(Y_train, [75,1])
Y_test = np.reshape(Y_test, [25,1])

X = np.reshape(X, [100,1])
Y = np.reshape(Y, [100,1])

n_units = 1
n_inputs = 1
n_outputs = 1

x = tf.placeholder(tf.float32,[None,n_inputs])
y = tf.placeholder(tf.float32,[None,n_outputs])

m = tf.Variable(0.5)
b = tf.Variable(0.5)

z = tf.add(tf.multiply(x,m),b)

loss = tf.losses.mean_squared_error(labels = y, predictions = z)
gdo = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

epoch = 1000

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    loss_list = []
    for e in range(epoch+1):
        c = sess.run([gdo,loss],feed_dict = {x:X_train, y:Y_train})
        loss_list.append(c[1])
        if (e % 100 == 0):
            print("Epoch ",e)
            print("Loss ",c[1])

    Y_pred = z.eval({x:X_test})
    Z = z.eval({x:X})
    sess.close()

plt.plot(loss_list)
plt.title('Loss function')
plt.legend()
plt.show()

plt.scatter(X, Y)
plt.plot(X, Z, c = 'red') 
plt.show()

d1 = Y_test - Y_pred
d2 = Y_test - Y_test.mean()
Rsq = 1 - d1.T.dot(d1) / d2.T.dot(d2)
print('R-square value = ',Rsq[0][0]*100)





