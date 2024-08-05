import numpy as np
import pandas as pd
from tensorflow import keras
import statistics


def build_model(sequence, act_functions):
    """
    Builds a sequential neural network model with the specified architecture.

    Args:
        sequence (list): A list of integers representing the number of units in each dense layer.
        act_functions (list): A list of activation functions corresponding to each dense layer.

    Returns:
        keras.Sequential: The compiled sequential model.

    Example:
        # Build a model with two dense layers
        sequence = [64, 32, 10]
        act_functions = ['relu', 'relu', 'softmax']
        model = build_model(sequence, act_functions)

    """
    model = keras.Sequential()

    # Add dense layers with specified units and activation functions
    for i, act_function in enumerate(act_functions):
        model.add(keras.layers.Dense(sequence[i], activation=act_function))

    # Compile the model with optimizer, loss, and metrics
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


#   loading data
fashion_mnist = keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist

#   splitting data
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid_fashion = X_train_full[-5000:], y_train_full[-5000:]

#   preparing data
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
num_output = np.unique(y_train).size

print('###   NOTICE: it takes a long time to run,'
      ' you can comment some config or reduce epochs to reduce running time.'
      ' the accuracy will reduce correspondingly  ###')
epochs = 10

#   setting different configurations
config_list = [[[546, 546, num_output], ['relu', 'softmax']],
               [[128, 128, num_output], ['relu', 'softmax']],
               [[64, 64, num_output], ['relu', 'softmax']],
               [[546, 128, num_output], ['relu', 'softmax']],
               [[128, 64, num_output], ['relu', 'softmax']],
               [[64, 32, num_output], ['relu', 'softmax']],
               [[546, 546, num_output], ['relu', 'sigmoid']],
               [[128, 128, num_output], ['sigmoid', 'sigmoid']],
               [[64, 64, num_output], ['sigmoid', 'sigmoid']],
               [[546, 128, num_output], ['relu', 'sigmoid']],
               [[128, 64, num_output], ['relu', 'sigmoid']],
               [[64, 32, num_output], ['relu', 'sigmoid']],
               ]

#   training and testing model for each configuration and keeping test results
models, train_accuracy, test_accuracy, valid_accuracy = [], [], [], []
for i, config in enumerate(config_list):
    #   training
    models.append(build_model(config[0], config[1]))
    history = models[i].fit(X_train, y_train, epochs=epochs, batch_size=128)

    #   evaluating
    loss1, test_acc, = models[i].evaluate(X_test, y_test)
    loss2, valid_acc, = models[i].evaluate(X_valid, y_valid_fashion)

    #   keeping tests result
    train_accuracy.append(statistics.mean(history.history['accuracy']))
    test_accuracy.append(test_acc)
    valid_accuracy.append(valid_acc)

#   printing test results

print(f'\n###  Configurations:  ###')
for i, config in enumerate(config_list):
    print(f'Model {i + 1}:    {config}')

data = {
    'model': np.arange(0, len(models), 1),
    'Train Accuracy': train_accuracy,
    'Valid Accuracy': valid_accuracy,
    'Test Accuracy': test_accuracy
}
df = pd.DataFrame(data)
print(df)
print(f'\nBest in training data: model {np.argmax(train_accuracy) + 1}.')
print(f'Best in validation data: model {np.argmax(valid_accuracy) + 1}.')
print(f'Best in test data: model {np.argmax(test_accuracy) + 1}.')
