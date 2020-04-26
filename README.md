# Solar_Prediction-model


INRODUCTION TO THE PROBLEM: 

With the increase in the consumption of renewable energy resources like solar and wind power,their unpredictability leads to a lot of inefficiency in the electric grid.If we know the power output of the renewable energy source before hand, we can efficiently design the operating of the grid.

In this project we deal with predicting the solar irradiance based on a set of 9 parameters,which include: Temperature,Pressure,Humidity,Wind direction, Wind speed ,length of the day, hour of the day, month and year.

A dataset has been downloaded from the internet which has recorded data of the solar irradiance measured for 4 consecutive years .We split this data into training and test sets.


IMPLEMENTATION OF THE NEURAL NETWORK MODEL:

I used a 3 layer neural network,the first layer has 9  neurons(the input features), the second layer has 4 neurons(the hidden layer) and the output layer consists of a single neuron.

I used RELU as the activation function,as it gave better results than the sigmoid activation function

I have defined my own cost function and gradient functions using the backpropogation algorithm.The cost function and the gradient function are passed to the scipy.optimise.fmin_tnc() function which performs stoichastic gradient descent to find the optimum weights for the neural network through a 30 epochs.

RESULTS:

After performing a parametric sweep over the regularisation term :lamda, I have found out that lamda=0.001 gives the best results.

Using this value of lamda, training the neural network with the training data, and using it upon the test data it results in a mean error of 3.57 over the test data.The standard deviation of the error is around 2.65.

The neural network could still be improved and I am still working on it, before i get a chance to verify the working of the code with real time data from our solar lab, once our college reopens.

