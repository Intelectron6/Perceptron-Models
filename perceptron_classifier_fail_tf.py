#Single Layered Perceptron Classifier using Tensorflow to solve Donut problem

import sys

if not sys.warnoptions:
    import warnings, os
    warnings.simplefilter("ignore")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 500
D = 2

R_inner = 6
R_outer = 9

R1 = np.random.randn(N//2) + R_inner
theta = 2*np.pi*np.random.random(N//2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N//2) + R_outer
theta = 2*np.pi*np.random.random(N//2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([ X_inner, X_outer ])
Y = np.array([0]*(N//2) + [1]*(N//2))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Y_train = np.reshape(Y_train, [375,1])
Y_test = np.reshape(Y_test, [125,1])

plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()

n_units = 1
n_inputs = 2
n_outputs = 1

x = tf.placeholder(tf.float32,[None,n_inputs])
y = tf.placeholder(tf.float32,[None,n_outputs])

w = tf.Variable(tf.random_normal([n_inputs,n_units]))
b = tf.Variable(tf.random_normal([n_units]))

z = tf.sigmoid(tf.add(tf.matmul(x,w),b))

loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = y, logits = z)
adamo = tf.train.AdamOptimizer(learning_rate = 0.2).minimize(loss)

epoch = 1000

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    loss_list = []
    for e in range(epoch + 1):
        c = sess.run([adamo,loss],feed_dict = {x:X_train, y:Y_train})
        loss_list.append(c[1])
        if (e % 100 == 0):
            print("Epoch ",e)
            print("Loss ",c[1])

    Y_pred = z.eval({x:X_test})
    sess.close()

plt.plot(loss_list)
plt.title('Loss function')
plt.legend()
plt.show()

Y_pred = np.round(Y_pred)
right_predictions = 0

for i in range(len(Y_pred)):
    if (Y_pred[i] == Y_test[i]):
        right_predictions += 1
accuracy = (right_predictions/len(Y_test))*100
print("Test accuracy = ", accuracy)

if (accuracy < 75):
    print("Perceptron can't solve a donut problem")
else:
    print("Fluke! Try again, the accuracy will most likely be less")


