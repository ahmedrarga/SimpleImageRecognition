from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import numpy as np
import random
import cv2
import os


def load_images(path, size, flatten):
    '''
    loading images from folder
    :param path: the path of the images
    :param size: size of images to set
    :param flatten: is Flatten?
    :return: tuple of numpy arrays: images and labels
    '''
    print("loading images from " + path + "...")
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


def initialize_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(3, activation="softmax"))
    return model

learning_rate = 0.01
epochs = 50
batch_size = 32


def task1():
    print("evaluating network on lab3-data ...")
    data, labels = load_images(path="lab3-data", size=(32, 32), flatten=True)
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    print("training network...")
    opt = SGD(lr=learning_rate)
    model = initialize_model()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=batch_size)
    predictions = model.predict(testX, batch_size=batch_size)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
    print("predicting our images")
    print("loading my images...")
    data_2 = []
    labels_2 = []
    imagePaths = sorted(list(paths.list_images("my_images")))
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        image = image.reshape((1, image.shape[0]))

        preds = model.predict(image, batch_size=1)
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        print("{}, {}".format(imagePath, text))
    print("==================================================================")


def task2():
    print("evaluating network on lab3-data and our data ...")
    data, labels = load_images(path="lab3-data", size=(32, 32), flatten=True)
    trainX, trainY = data, labels
    testX, testY = load_images(path="my_Images", size=(32,32), flatten=True)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    print("training network...")
    opt = SGD(lr=learning_rate)
    model = initialize_model()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=batch_size)
    predictions = model.predict(testX, batch_size=batch_size)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
    print("predicting our images")
    print("loading my images...")
    data_2 = []
    labels_2 = []
    imagePaths = sorted(list(paths.list_images("my_images")))
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        image = image.reshape((1, image.shape[0]))

        preds = model.predict(image, batch_size=1)
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        print("{}, {}".format(imagePath, text))

print("Put All Data in one directory, each data directory must have the classess to classify to them")
data_path = input("Enter path of data: ")
task1(data_path)
task2(data_path)