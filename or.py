#OR GATE implementation using Single Layered Perceptron 

import numpy as np
import random

# function to predict output value
# if calculated value is less than 0, predict 0
# if calculated value is greater than or equal to 0, predict 1
def predict(X, w, b):
	p =  np.sign(X.dot(w) + b) 
	if p == -1:		    	
		p = 0
	else:
		p = 1
	return p

# function to train the perceptron
def fit(learning_rate, epoch):
						
	#initialize weights and bias
	w = np.array([1,1])
	b = -1
	
	#loop to update weights based on predictions and actual value
	for e in range(epoch):	
		X1 = random.choice([0,1])
		X2 = random.choice([0,1])
		X = np.array([X1,X2])
		Y = X1 or X2
		Yp = predict(X, w, b)
		if Y == Yp:
			continue		
		else:
			w = w - learning_rate*(X.dot(Yp-Y))
			b = b - learning_rate*(Yp-Y)
	return w, b

#initialize learning rate and epoch
learning_rate = 0.499
epoch = 100

#call the fit function for training 
w, b = fit(learning_rate, epoch)

print("w1 = ",w[0], "; w2 = ",w[1],"; b = ",b)

print("X1  X2  Y")
X1 = [0,0,1,1]
X2 = [0,1,0,1]
for i in range(4):
	X = np.array([X1[i],X2[i]])
	print(X1[i]," ",X2[i]," ",predict(X, w, b)) 

