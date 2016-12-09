import numpy as np
import pickle
import tensorflow as tf
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Flatten

flags = tf.app.flags
FLAGS = flags.FLAGS

# Here we define some command line flags, this avoids having to manually open and edit
# the file if we want to change the files we train and validate our model with.

# Here's how you would run the file from the command line:
#
# python feature_extraction.py  --training_file vgg_cifar10_100_bottleneck_features_train.p --validation_file vgg_cifar10_bottleneck_features_validation.p --epochs 5 --batch_size 128 --learning_rate 0.001

# Running this program will train feature extraction with the VGG network/Cifar10 dataset bottleneck features.
# The 100 in vgg_cifar10_100 indicates this file has 100 examples per class.

# You could define additional flags if you wish. Possible candidates could be the batch size or the number of epochs.

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('batch_size', 256, "Number of items to sample during mini batch operations")
flags.DEFINE_integer('epochs', 50, "Max number of iterations to train the network")
flags.DEFINE_float('learning_rate', 1e-3, "How large of a step we should allow our network to take at each batch")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    # TODO: train your model here

    nb_classes = len(np.unique(y_train))

    # define model
    input_shape = X_train.shape[1:]
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val),
              shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
