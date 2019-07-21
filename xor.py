#XOR GATE implementation using Combination of Perceptrons 

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
	w1 = np.array([1,1])
	b1 = -1
	w2 = np.array([1,1])
	b2 = -1
	w3 = np.array([1,1])
	b3 = -1
	
	#loop to update weights based on predictions and actual value
	for e in range(epoch):	
		X1 = random.choice([0,1])
		X2 = random.choice([0,1])
		Y1 = X1 or X2
		Y2 = not (X1 and X2)
		Y = Y1 and Y2
		Xa = np.array([X1,X2])
		Y1p = predict(Xa, w1, b1)
		Y2p = predict(Xa, w2, b2)
		X = np.array([Y1p,Y2p])
		Yp = predict(X, w3, b3)
		if Y1p == Y1:
			pass		
		else:
			w1 = w1 - learning_rate*(Xa.dot(Y1p-Y1))
			b1 = b1 - learning_rate*(Y1p-Y1)
		if Y2p == Y2:
			pass		
		else:
			w2 = w2 - learning_rate*(Xa.dot(Y2p-Y2))
			b2 = b2 - learning_rate*(Y2p-Y2)
		if Y == Yp:
			continue		
		else:
			w3 = w3 - learning_rate*(X.dot(Yp-Y))
			b3 = b3 - learning_rate*(Yp-Y)
	return w1, b1, w2, b2, w3, b3

#initialize learning rate and epoch
learning_rate = 0.499
epoch = 100

#call the fit function for training 
w1, b1, w2, b2, w3, b3 = fit(learning_rate, epoch)

print("Weights for OR - w1 = ",w1[0], "; w2 = ",w1[1],"; b = ",b1)
print("Weights for NAND - w1 = ",w2[0], "; w2 = ",w2[1],"; b = ",b2)
print("Weights for AND  - w1 = ",w3[0], "; w2 = ",w3[1],"; b = ",b3)

print("X1  X2  Y")
X1 = [0,0,1,1]
X2 = [0,1,0,1]
for i in range(4):
	Xa = np.array([X1[i],X2[i]])
	Y1p = predict(Xa, w1, b1)
	Y2p = predict(Xa, w2, b2)
	X = np.array([Y1p,Y2p])
	Yp = predict(X, w3, b3)
	print(X1[i]," ",X2[i]," ",Yp) 
