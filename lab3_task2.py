from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from sklearn.metrics import classification_report
import random
from imutils import paths
import cv2
import numpy as np
import os

def load_images(path, size, flatten):
    print("loading images...")
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        if flatten:
            image = cv2.resize(image, size).flatten()
        else:
            image = cv2.resize(image, size)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return (data, labels)


def Layer1(model):
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape(64, 64)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    return model


def Layer2(model):
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Dropout(0.25))
    return model


def Layer3(model):
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Dropout(0.25))
    return model


def final_layer(model):
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation("softmax"))
    return model


def input_shape(width, height):
    return (height, width, 3)


chanDim = -1
learning_rate = 0.01
epochs = 75
batch_size = 1


def task3():
    data, labels = load_images(path="10photos-data", size=(64,64), flatten=False)
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

    model = Sequential()
    model = Layer1(model)
    model = Layer2(model)
    model = Layer3(model)
    model = final_layer(model)
    print("training network...")
    opt = SGD(lr=learning_rate, decay=learning_rate / epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size, epochs=epochs)

    print("evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=lb.classes_))

def task4():
    data, labels = load_images(path="10photos-data", size=(64, 64), flatten=False)
    trainX, trainY = data, labels
    testX, testY = load_images("my_images", (64,64), flatten=False)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    model = Sequential()
    model = Layer1(model)
    model = Layer2(model)
    model = Layer3(model)
    model = final_layer(model)
    print("training network...")
    opt = SGD(lr=learning_rate, decay=learning_rate / epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size, epochs=epochs)

    print("evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lb.classes_))

task3()
task4()