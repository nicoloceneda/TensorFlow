""" REGRESSION - TENSORFLOW
    -----------------------
    Execution of the code at: https://www.tensorflow.org/tutorials/keras/regression
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


# Import libraries

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Visualization settings
np.set_printoptions(precision=3, edgeitems=10, suppress=True)
pd.set_option('max_columns', 10, 'max_rows', 10)


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the data

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()

print('='*88, '{:^88}'.format('DATASET'), '='*88, dataset.tail(), sep='\n')


# Remove rows with missing values

print('='*18, '{:^18}'.format('N/A VALUES'), '='*18, dataset.isna().sum(), sep='\n')

dataset = dataset.dropna()


# One-hot encode the categorical variable ('Origin')

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

print('='*101, '{:^101}'.format('ONE HOT ENCODING'), '='*101, dataset.tail(), sep='\n')


# Split the data into train and test sets

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# Visually inspect the train set

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind='kde')


# Inspect the statistics of the train set

train_stats = train_dataset.describe().transpose()

print('='*87, '{:^87}'.format('STATISTICS OF THE TRAIN DATASET'), '='*87, train_stats, sep='\n')


# Create train and test sets of features (for models with one and multiple predictors) and labels

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

train_features_2 = np.array(train_features['Horsepower'])
test_features_2 = np.array(test_features['Horsepower'])


# -------------------------------------------------------------------------------
# 2. LINEAR REGRESSION WITH ONE FEATURE
# -------------------------------------------------------------------------------


# Create the model

layer_0 = tf.keras.layers.Normalization(input_shape=(1,), axis=None, name='Normalizer')
layer_0.adapt(train_features_2)

layer_1 = tf.keras.layers.Dense(units=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='Linear')

model = tf.keras.Sequential([layer_0, layer_1])
model.summary()


# Configure the training procedure

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')


# Train the model

training = model.fit(x=train_features_2, y=train_labels, batch_size=None, epochs=100, verbose=1, validation_split=0.2)


# Visualize the training progress

training_history = pd.DataFrame(training.history)
training_history['epoch'] = training.epoch

print('='*29, '{:^29}'.format('TRAINING HISTORY'), '='*29, training_history.tail(), sep='\n')


# Plot the training history

plt.figure()
plt.plot(training.history['loss'], label='loss')
plt.plot(training.history['val_loss'], label='val_loss')
plt.ylim([0,10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)


# Test the model

testing_loss = model.evaluate(x=test_features_2, y=test_labels, batch_size=None, verbose=1)
test_results = {'one_feature': testing_loss}

print('='*17, '{:^17}'.format('TESTING LOSS'), '='*17, testing_loss, sep='\n')


# Since this is a single feature regression, plot the model's predictions as a function of the input

x = tf.linspace(0.0, 250, 251)
y = model.predict(x)

plt.figure()
plt.scatter(train_features_2, train_labels, label='Data')
plt.plot(x, y, color='black', label='Predictions')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.grid(True)


# -------------------------------------------------------------------------------
# 3. LINEAR REGRESSION WITH MULTIPLE FEATURES
# -------------------------------------------------------------------------------


# Create the model

layer_0 = tf.keras.layers.Normalization(input_shape=(9,), axis=1, name='Normalizer')
layer_0.adapt(train_features)

layer_1 = tf.keras.layers.Dense(units=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='Linear')

model = tf.keras.Sequential([layer_0, layer_1])
model.summary()


# Configure the training procedure

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')


# Train the model

training = model.fit(x=train_features, y=train_labels, batch_size=None, epochs=100, verbose=1, validation_split=0.2)


# Visualize the training progress

training_history = pd.DataFrame(training.history)
training_history['epoch'] = training.epoch

print('='*29, '{:^29}'.format('TRAINING HISTORY'), '='*29, training_history.tail(), sep='\n')


# Plot the training history

plt.figure()
plt.plot(training.history['loss'], label='loss')
plt.plot(training.history['val_loss'], label='val_loss')
plt.ylim([0,10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)


# Test the model

testing_loss = model.evaluate(x=test_features, y=test_labels, batch_size=None, verbose=1)
test_results['multiple_features'] = testing_loss

print('='*17, '{:^17}'.format('TESTING LOSS'), '='*17, testing_loss, sep='\n')


# Make predictions

test_predictions = model.predict(test_features)

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.plot([0,50], [0,50])
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.xlim([0,50])
plt.ylim([0,50])


# Error distribution

error = test_predictions.flatten() - test_labels

plt.figure()
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
plt.ylabel('Count')


# -------------------------------------------------------------------------------
# 4. LINEAR REGRESSION WITH A DEEP NEURAL NETWORK AND ONE FEATURE
# -------------------------------------------------------------------------------


# Create the model

layer_0 = tf.keras.layers.Normalization(input_shape=(1,), axis=None, name='Normalizer')
layer_0.adapt(train_features_2)

layer_1 = tf.keras.layers.Dense(units=64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='Non_Linear_1')
layer_2 = tf.keras.layers.Dense(units=64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='Non_Linear_2')
layer_3 = tf.keras.layers.Dense(units=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='Linear')

model = tf.keras.Sequential([layer_0, layer_1, layer_2, layer_3])
model.summary()


# Configure the training procedure

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')


# Train the model

training = model.fit(x=train_features_2, y=train_labels, batch_size=None, epochs=100, verbose=1, validation_split=0.2)


# Visualize the training progress

training_history = pd.DataFrame(training.history)
training_history['epoch'] = training.epoch

print('='*29, '{:^29}'.format('TRAINING HISTORY'), '='*29, training_history.tail(), sep='\n')


# Plot the training history

plt.figure()
plt.plot(training.history['loss'], label='loss')
plt.plot(training.history['val_loss'], label='val_loss')
plt.ylim([0,10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)


# Test the model

testing_loss = model.evaluate(x=test_features_2, y=test_labels, batch_size=None, verbose=1)
test_results['one_feature_dnn'] = testing_loss

print('='*17, '{:^17}'.format('TESTING LOSS'), '='*17, testing_loss, sep='\n')


# Since this is a single feature regression, plot the model's predictions as a function of the input

x = tf.linspace(0.0, 250, 251)
y = model.predict(x)

plt.figure()
plt.scatter(train_features_2, train_labels, label='Data')
plt.plot(x, y, color='black', label='Predictions')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.grid(True)


# -------------------------------------------------------------------------------
# 5. LINEAR REGRESSION WITH A DEEP NEURAL NETWORK AND MULTIPLE FEATURES
# -------------------------------------------------------------------------------


# Create the model

layer_0 = tf.keras.layers.Normalization(input_shape=(9,), axis=1, name='Normalizer')
layer_0.adapt(train_features)

layer_1 = tf.keras.layers.Dense(units=64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='Non_Linear_1')
layer_2 = tf.keras.layers.Dense(units=64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='Non_Linear_2')
layer_3 = tf.keras.layers.Dense(units=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='Linear')

model = tf.keras.Sequential([layer_0, layer_1, layer_2, layer_3])
model.summary()


# Configure the training procedure

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')


# Train the model

training = model.fit(x=train_features, y=train_labels, batch_size=None, epochs=100, verbose=1, validation_split=0.2)


# Visualize the training progress

training_history = pd.DataFrame(training.history)
training_history['epoch'] = training.epoch

print('='*29, '{:^29}'.format('TRAINING HISTORY'), '='*29, training_history.tail(), sep='\n')


# Plot the training history

plt.figure()
plt.plot(training.history['loss'], label='loss')
plt.plot(training.history['val_loss'], label='val_loss')
plt.ylim([0,10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)


# Test the model

testing_loss = model.evaluate(x=test_features, y=test_labels, batch_size=None, verbose=1)
test_results['multiple_features_dnn'] = testing_loss

print('='*17, '{:^17}'.format('TESTING LOSS'), '='*17, testing_loss, sep='\n')


# Make predictions

test_predictions = model.predict(test_features)

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.plot([0,50], [0,50])
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.xlim([0,50])
plt.ylim([0,50])


# Error distribution

error = test_predictions.flatten() - test_labels

plt.figure()
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
plt.ylabel('Count')


# -------------------------------------------------------------------------------
# 6. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()