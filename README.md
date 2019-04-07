# Project Docs

## Project Setup

For this project I have decided to use [TensorFlow](https://tensorflow.org) due to the ease of use of its [keras api](https://www.tensorflow.org/api_docs/python/tf/keras).

I started by splitting the sample data set into a training data set and a testing data set. I decided to use 17% of the total sample for testing and the rest for training.

## Optimizer

From the keras api I stuck with these two optimizers. 

The first one, was the Adam optimizer. I tried 3 learning rates for it, 0.001, 0.0005 and 0.00146. Any value above 0.0015 seemed to increase the learning time and hurt the accuracy.

The second one, was the <abbr title="Stochastic Gradient Descent">SGD</abbr> optimizer. During testing, this one returned a loss closer to the one during the training process (about 0.13 during testing and about 0.10 during training).

I chose to stick to these two optimizers because they were the fastest compared to the ones I have tested (especially Adam with its default learning rate). 

## Loss

For the loss function I chose the __Sparse Categorical Crossentropy__ function because it is the most recommended function when it comes to doing labeling or classification. I have also tried the __Kullback Leibler Divergence__ loss function and some other variations of the __Categorical Crossentropy__, but the <abbr title="Sparse Categorical Crossentropy">SCC</abbr> yield the best results.

## Layers

For both configurations I used a neural network that consists of 4 layers. 

The first one is flattening the input into a one dimensional array.

The next two layers are densely connected. Each neuron takes input from all of the neurons in the previous layer, weighting that input and outputs a single value to the next layer.

The last layer, takes input from the previous neurons and outputs a value in between 0 and 1 which represents the probability that the given input is associated to that label. I chose 8 neurons for this layer because there are 8 labels in total.

## Activation Functions

For the activation functions I chose <abbr title="rectified linear">relu</abbr> and <abbr title="scaled exponential linear">selu</abbr> simply because they are the most common transfer functions. However, I chose softmax because it is one of the best functions at doing classifications.

## Training the model

I went with only 7 epochs because the optimizers I used had a pretty fast learning rate and I wanted to avoid overfitting the model.

I also wanted to experiment with repeating, shuffling and batching the training data set, but I had only managed to do so with datasets built within tensorflow, and not with the provided ones.
