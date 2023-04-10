import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from scipy.io import loadmat        # to load mat files
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import cv2                          # for colour conversion
import tensorflow as tf             # for bulk image resize
import keras
from keras import layers
from tensorboard import notebook
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from time import process_time
from tensorboard import program


# Load data for Q3
#  train_path: path to training data mat file
#  test_path:  path to testing data mat file
#
#  returns:    arrays for training and testing X and Y data
#
def load_data(train_path, test_path):

    # load files
    train = loadmat(train_path)
    test = loadmat(test_path)

    # transpose, such that dimensions are (sample, width, height, channels), and divide by 255.0
    train_X = np.transpose(train['train_X'], (3, 0, 1, 2)) / 255.0
    train_Y = train['train_Y']
    # change labels '10' to '0' for compatability with keras/tf. The label '10' denotes the digit '0'
    train_Y[train_Y == 10] = 0
    train_Y = np.reshape(train_Y, -1)

    # transpose, such that dimensions are (sample, width, height, channels), and divide by 255.0
    test_X = np.transpose(test['test_X'], (3, 0, 1, 2)) / 255.0
    test_Y = test['test_Y']
    # change labels '10' to '0' for compatability with keras/tf. The label '10' denotes the digit '0'
    test_Y[test_Y == 10] = 0
    test_Y = np.reshape(test_Y, -1)

    # return loaded data
    return train_X, train_Y, test_X, test_Y

# vectorise an array of images, such that the shape is changed from {samples, width, height, channels} to
# (samples, width * height * channels)
#   images: array of images to vectorise
#
#   returns: vectorised array of images
#
def vectorise(images):
    # use numpy's reshape to vectorise the data
    return np.reshape(images, [len(images), -1])

# Plot some images and their labels. Will plot the first 100 samples in a 10x10 grid
#  x: array of images, of shape (samples, width, height, channels)
#  y: labels of the images
#
def plot_images(x, y):
    fig = plt.figure(figsize=[15, 18])
    for i in range(100):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.imshow(x[i,:])
        ax.set_title(y[i])
        ax.axis('off')

# Resize an array of images
#  images:   array of images, of shape (samples, width, height, channels)
#  new_size: tuple of the new size, (new_width, new_height)
#
#  returns:  resized array of images, (samples, new_width, new_height, channels)
#
def resize(images, new_size):
    # tensorflow has an image resize funtion that can do this in bulk
    # note the conversion back to numpy after the resize
    return tf.image.resize(images, new_size).numpy()
          
# Convert images to grayscale
#   images:  array of colour images to convert, of size (samples, width, height, 3)
#
#   returns: array of converted images, of size (samples, width, height, 1)
#
def convert_to_grayscale(images):
    # storage for converted images
    gray = []
    # loop through images
    for i in range(len(images)):
        # convert each image using openCV
        gray.append(cv2.cvtColor(images[i,:], cv2.COLOR_BGR2GRAY))
    # pack converted list as an array and return
    return np.expand_dims(np.array(gray), axis = -1)

def eval_model(model, x_test, y_test):
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)
    i = tf.cast([], tf.int32)
    indexes = tf.gather_nd(indexes, i)
    
    cm = confusion_matrix(y_test, indexes)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)

    print(classification_report(y_test, indexes))

# load data
train_X, train_Y, test_X, test_Y = load_data('Data/q3_train.mat', 'Data/q3_test.mat')

# any resize, colour change, etc, would go here

data_augmentation = keras.Sequential([
  #layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.025),
  layers.RandomZoom(0.025),
  layers.RandomTranslation(height_factor=(-0.02, 0.02), width_factor=(-0.02, 0.02))
])


# Basic DCNN Model
# our model, input again, still in an image shape
inputs = keras.Input(shape=(32, 32, 3, ), name='img')
augmented = data_augmentation(inputs)
# 3x3 conv block, we have two conv layers, and a max-pooling. The conv layers have identical parameters
# and are simply separated by an activation, in our case, relu
x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation=None)(augmented)
x = layers.Activation('relu')(x)
x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation=None)(x)
# batch normalisation, before the non-linearity
x = layers.BatchNormalization()(x)
# spatial dropout, this will drop whole kernels, i.e. 20% of our 3x3 filters will be dropped out rather
# than dropping out 20% of the invidual pixels
x = layers.SpatialDropout2D(0.2)(x)
x = layers.Activation('relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

# 3x3 conv block, we have two conv layers, and a max-pooling. The conv layers have identical parameters
# and are simply separated by an activation, in our case, relu
x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation=None)(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation=None)(x)
# batch normalisation, before the non-linearity
x = layers.BatchNormalization()(x)
# spatial dropout, this will drop whole kernels, i.e. 20% of our 3x3 filters will be dropped out rather
# than dropping out 20% of the invidual pixels
x = layers.SpatialDropout2D(0.2)(x)
x = layers.Activation('relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

# 3x3 conv block, we have two conv layers, and a max-pooling. The conv layers have identical parameters
# and are simply separated by an activation, in our case, relu
x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=None)(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=None)(x)
# batch normalisation, before the non-linearity
x = layers.BatchNormalization()(x)
# spatial dropout, this will drop whole kernels, i.e. 20% of our 3x3 filters will be dropped out rather
# than dropping out 20% of the invidual pixels
x = layers.SpatialDropout2D(0.2)(x)
x = layers.Activation('relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

# 3x3 conv block, increase filters, same structure as above, but now with 16 filters
x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
# batch normalisation, before the non-linearity
x = layers.BatchNormalization()(x)
# spatial dropout, this will drop whole kernels, i.e. 20% of our 3x3 filters will be dropped out rather
# than dropping out 20% of the invidual pixels
x = layers.SpatialDropout2D(0.2)(x)
x = layers.Activation('relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

# 3x3 conv block, further increase filters to 32, again the structure is the same
x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=None)(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=None)(x)
# batch normalisation, before the non-linearity
x = layers.BatchNormalization()(x)
# spatial dropout, this will drop whole kernels, i.e. 20% of our 3x3 filters will be dropped out rather
# than dropping out 20% of the invidual pixels
x = layers.SpatialDropout2D(0.2)(x)
x = layers.Activation('relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

# flatten layer
x = layers.Flatten()(x)

# dense layer, 512 neurons
x = layers.Dense(512, activation='relu')(x)

# the output, 10 neurons for 10 classes, and a softmax activation
outputs = layers.Dense(10, activation='softmax')(x)

# build the model, and print a summary
model_vgg = keras.Model(inputs=inputs, outputs=outputs, name='vgg_for_cifar10')
model_vgg.summary()

model_vgg.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
history = model_vgg.fit(train_X, train_Y,
                        batch_size=125,
                        epochs=200,
                        validation_split=0.1)
eval_model(model_vgg, test_X, test_Y)

fig = plt.figure(figsize=[20, 6])
ax = fig.add_subplot(1, 2, 1)
ax.plot(history.history['loss'], label="Training Loss")
ax.plot(history.history['val_loss'], label="Validation Loss")
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.plot(history.history['accuracy'], label="Training Accuracy")
ax.plot(history.history['val_accuracy'], label="Validation Accuracy")
ax.legend()

plt.show()