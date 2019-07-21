#NOT GATE implementation using Single Layered Perceptron 

import numpy as np
import random

# function to predict output value
# if calculated value is less than 0, predict 0
# if calculated value is greater than or equal to 0, predict 1
def predict(X, w, b):
	p = np.sign(X*w + b)
	if p == -1:		    	
		p = 0
	else:
		p = 1
	return p

# function to train the perceptron
def fit(learning_rate, epoch):
						
	#initialize weights and bias
	w = np.array([1])
	b = -1
	
	#loop to update weights based on predictions and actual value
	for e in range(epoch):	
		X = random.choice([0,1])
		Y = not X
		Yp = predict(X, w, b)
		if Y == Yp:
			continue		
		else:
			w = w - learning_rate*(X*(Yp-Y))
			b = b - learning_rate*(Yp-Y)
	return w, b

#initialize learning rate and epoch
learning_rate = 0.499
epoch = 100

#call the fit function for training 
w, b = fit(learning_rate, epoch)

print("w = ",w, "; b = ",b)

print("X  Y")
print(0," ",predict(0, w, b))
print(1," ",predict(1, w, b))

