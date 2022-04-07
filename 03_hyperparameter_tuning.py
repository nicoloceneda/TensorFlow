""" HYPERPARAMETER TUNING - TENSORFLOW
    ----------------------------------
    Execution of the code at: https://www.tensorflow.org/tutorials/keras/keras_tuner
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


# Import libraries

import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras

# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the data

(img_train, label_train), (img_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()


# Normalize pixel values between 0 and 1

img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0


# -------------------------------------------------------------------------------
# 2. DEFINE THE MODEL
# -------------------------------------------------------------------------------


# Define the hypermodel using a model builder function

def model_builder(hp):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer: choose an optimal value between 32-512

    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    # Tune the learning rate for the optimizer: choose an optimal value from 0.01, 0.001, or 0.0001

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


# -------------------------------------------------------------------------------
# 3. PERFORM HYPERPARAMETER TUNING
# -------------------------------------------------------------------------------


# Instantiate the tuner to perform the hyperparameter tuning

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='hypertuning',
                     project_name='intro_to_kt')


# Create a callback to stop training early after reaching a certain value for the validation loss

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# Run the hyperparameter search

tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])


# Get the optimal hyperparameters

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is {} and the optimal "
      "learning rate for the optimizer is {}.".format(best_hps.get('units'), best_hps.get('learning_rate')))


# -------------------------------------------------------------------------------
# 4. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Build the model with the optimal hyperparameters and train it on the data for 50 epochs

model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


# Re-instantiate the hypermodel and train it with the optimal number of epochs from above

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)


# -------------------------------------------------------------------------------
# 4. TEST THE MODEL
# -------------------------------------------------------------------------------


# Evaluate the hypermodel on the test data

eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)