##import the classes from keras
from keras.models import Sequential
from keras.layers import Dense
##import fundamental package for scientific computing with Python
import numpy

# fix random seed for reproducibility
## this will give us a random numbers for global values
seed = 7
numpy.random.seed(seed)

##delimiter "," The string used to separate values. By default, this is any whitespace.
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

## use spacify the part of the matrix that we will use in the next calculations
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

##first input: number of neurons 
##second input: the number of model inputs
##init='uniform': parameter settings connected with the wire setting for intialization.
##activation='relu': activation functions that transform a summed signal from each neuron in a layer can be extracted and added to the Sequential as a layer-like object called Activation##
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


##loss: the loss function used to evaluate the network that is minimized by the optimization algorithm##
##optimizer:optimization algorithm for training the model.specifying the stochastic gradient descent (sgd) optimization algorithm ##
##metrics:specify metrics to collect while fitting your model in addition to the loss function. Generally, the most useful additional metric to collect is accuracy for classification problems. The metrics ##
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


##epoch:Each epoch can be partitioned into groups of input-output pattern pairs called batches##
##batch_size:This define the number of patterns that the network is exposed to before the weights are updated within an epoch##
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

#This returns an array value after that we pike the first value.
# evaluate the model
scores = model.evaluate(X, Y)

## text formatting% (return, return)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


