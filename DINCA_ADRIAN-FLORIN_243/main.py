import tensorflow as tf
import numpy as np
import pandas as pd
import math

# Whether or not to test the loss on the given data set
IS_TESTING_LOSS = True

# How much of the given data set should be use for testing (as a percentage)
TESTING_PERCENTAGE = 17

# Read the training data sets
samples = pd.read_csv('./train_samples.csv', header=None).values
labels = pd.read_csv('./train_labels.csv', header=None).values

training_samples = []
training_labels = []

# When the data set is being tested
if(IS_TESTING_LOSS):
    # Split it into training and testing
    TRAINING_LIMIT = len(samples) - math.floor(len(labels) * TESTING_PERCENTAGE / 100)

    training_samples = samples[:TRAINING_LIMIT]
    testing_samples = samples[TRAINING_LIMIT + 1:len(samples)]

    training_labels = labels[:TRAINING_LIMIT]
    testing_labels = labels[TRAINING_LIMIT + 1:len(samples)]
else: 
    training_samples = samples
    training_labels = labels

# Setup the neural network for the model to use
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.selu),
    tf.keras.layers.Dense(8, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(training_samples, training_labels, epochs=7)

# Save the model so that it can be reused
model_name = 'test-loss' if (IS_TESTING_LOSS) else 'num'
model.save(f'{model_name}.4.model')

if(IS_TESTING_LOSS):
    # Evaluate model's loss and accuracy
    loss, accuracy = model.evaluate(testing_samples, testing_labels)
    exit(code=0) # Stop

# Read test samples
testing_samples = pd.read_csv('./test_samples.csv', header=None).values

# Make predictions
predictions = model.predict(testing_samples)

# Write predictions to csv
ids = []
labels = []

# For each prediction
for i in range(len(predictions)):
    ids.append(i + 1) # Give it an id
    labels.append(np.argmax(predictions[i])) # Choose the prediction with the highest probability

df = pd.DataFrame({'Id': ids, 'Prediction': labels})

df.to_csv("submission-4.new.csv", encoding='utf-8', index=False)
