import cv2
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed
from sklearn import model_selection, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import DistanceMetric, KNeighborsClassifier
import joblib
import pickle
import warnings
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


class Preprocessing:
    global img_directory
    global threshold
    global inv_threshold
    global image
    global preprocessed_digits
    global selectDigits
    global projection
    global sort_contours
    global initializeDataset
    global displayDatasetExample
    global projection_database
    global train
    global threshold_database
    global process
    global model
    global test

    def sort_contours(cnts):
        # initialize the reverse flag and sort index
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i]))

        return cnts

    def selectDigits(img_directory, thresh):
        image = cv2.imread(img_directory)

        _, contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sort_contours(contours)
        preprocessed_digits = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
            cv2.rectangle(image, (x, y), (x+w, y+h),
                          color=(0, 255, 0), thickness=2)

            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = thresh[y:y+h, x:x+w]

            # Resizing that digit to (18, 18)
            # print(digit.shape)
            digit_w, digit_h = digit.shape
            value = abs(digit_h - digit_w)
            digit = np.pad(digit, ((0, 0), (value//2, value//2)),
                           "constant", constant_values=0)

            resized_digit = cv2.resize(digit, (18, 18))

            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)),
                                  "constant", constant_values=0)

            # Adding the preprocessed digit to the list of preprocessed digits
            preprocessed_digits.append(padded_digit)

        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_img)

        plt.show()

        inp = np.array(preprocessed_digits)

        return inp

    def threshold(img_dir):
        global image
        original = cv2.imread(img_dir)
        # plt.imshow(image, None)
        # cv2.imshow("imaget", image)
        # cv2.waitKey(0)

        image = cv2.imread(img_dir, 0)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        ret, threshold = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_array = [blur, 0, threshold]
        return threshold, image_array

    def threshold_database(image):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        ret, threshold = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_array = [blur, 0, threshold]
        return threshold

    def inv_threshold(img_dir):
        image = cv2.imread(img_dir)
        # plt.imshow(image, None)
        # cv2.imshow("imaget", image)
        # cv2.waitKey(0)

        image = cv2.imread(img_dir, 0)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        ret, threshold = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        image_array = [blur, 0, threshold]
        return threshold, image_array

    def projection(preprocessed_digits):
        projectedDigits = []
        for digit in preprocessed_digits:

            vertical_p = digit.copy()
            horizontal_p = digit.copy()
            concat_p = digit.copy()
            (h, w) = vertical_p.shape
            (h, w) = horizontal_p.shape
            a = [0 for z in range(0, w)]
            b = [0 for z in range(0, w)]

            for j in range(0, w):  # Traversing a column
                for i in range(0, h):  # Traverse a row
                    if vertical_p[i, j] == 0:  # If you change the point to black
                        a[j] += 1  # Counter of this column plus one count
                        vertical_p[i, j] = 255  # Turn it white after recording

            for j in range(0, w):  # Traverse each column
                # Start from the top point of the column that should be blackened to the bottom
                for i in range((h-a[j]), h):
                    vertical_p[i, j] = 0  # Blackening
                    concat_p[i, j] = 0

            for j in range(0, h):
                for i in range(0, w):
                    if horizontal_p[j, i] == 0:
                        b[j] += 1
                        horizontal_p[j, i] = 255

            for j in range(0, h):
                for i in range(0, b[j]):
                    horizontal_p[j, i] = 0

            concat_p = vertical_p + horizontal_p

            fig = plt.figure(figsize=(10, 7))

            fig.add_subplot(1, 4, 1)
            plt.imshow(digit.reshape(28, 28), cmap="gray")
            plt.axis('off')
            plt.title("Digito")

            fig.add_subplot(1, 4, 2)
            plt.imshow(vertical_p, cmap=plt.gray())
            plt.title("Projecao Vertical")

            fig.add_subplot(1, 4, 3)
            plt.imshow(horizontal_p, cmap=plt.gray())
            plt.title("Projecao Horizontal")

            fig.add_subplot(1, 4, 4)
            plt.imshow(concat_p, cmap=plt.gray())
            plt.title("Projecao Concatenada")

            plt.show()
            projectedDigits.append(concat_p)
        return projectedDigits

    def projection_database(digit):
        vertical_p = digit.copy()
        horizontal_p = digit.copy()
        concat_p = digit.copy()
        (h, w, _) = vertical_p.shape
        (h, w, _) = horizontal_p.shape
        a = [0 for z in range(0, w)]
        b = [0 for z in range(0, w)]

        for j in range(0, w):  # Traversing a column
            for i in range(0, h):  # Traverse a row
                if vertical_p[i, j] == 0:  # If you change the point to black
                    a[j] += 1  # Counter of this column plus one count
                    vertical_p[i, j] = 255  # Turn it white after recording

        for j in range(0, w):  # Traverse each column
            # Start from the top point of the column that should be blackened to the bottom
            for i in range((h-a[j]), h):
                vertical_p[i, j] = 0  # Blackening
                concat_p[i, j] = 0

        for j in range(0, h):
            for i in range(0, w):
                if horizontal_p[j, i] == 0:
                    b[j] += 1
                    horizontal_p[j, i] = 255

        for j in range(0, h):
            for i in range(0, b[j]):
                horizontal_p[j, i] = 0

        concat_p = vertical_p + horizontal_p
        return concat_p

    def initializeDataset():
        # Let’s initialize the dataset and segregate into Training and Test set
        return mnist.load_data()

    def displayDatasetExample(X_train, y_train):
        # Now, just to get idea of what an image in the dataset looks like, let’s display it using matplotlib. Here’s the use of “import matplotlib.pyplot as plt”.
        plt.imshow(X_train[0], cmap="gray")
        plt.show()
        print(y_train[0])

    def process(i):
        return projection_database(i)

    def train():
        global preprocessedDigits
        global model

        mnist = datasets.load_digits()
        (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                          mnist.target, test_size=0.25, random_state=42)

        (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
                                                                        test_size=0.1, random_state=84)

        kVals = range(1, 30, 2)
        accuracies = []

        for k in range(1, 30, 2):
            # train the k-Nearest Neighbor classifier with the current value of `k`
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(trainData, trainLabels)

            # evaluate the model and update the accuracies list
            score = model.score(valData, valLabels)
            print("k=%d, accuracy=%.2f%%" % (k, score * 100))
            accuracies.append(score)

        i = int(np.argmax(accuracies))

        (X_train, y_train), (X_test, y_test) = initializeDataset()

        # So, let’s reshape the dataset according to our model.
        # 60000 -> number of images, 28x28 -> size of each image, 1 -> image in greyScale
        X_train = X_train.reshape(28, 28)
        X_test = X_test.reshape(28, 28)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        X_train_proj = []
        X_train_proj = Parallel(n_jobs=4)(
            delayed(process)(i) for i in X_train)

        X_test_proj = []
        X_test_proj = Parallel(n_jobs=4)(
            delayed(process)(i) for i in X_test)

        X_train = X_train_proj
        X_test = X_test_proj

        with open("X_train_database.txt", "wb") as fp:
            pickle.dump(X_train, fp)

        with open("y_train_database.txt", "wb") as fp:
            pickle.dump(y_train, fp)

        with open("X_test_database.txt", "wb") as fp:
            pickle.dump(X_test, fp)

        with open("y_test_database.txt", "wb") as fp:
            pickle.dump(y_test, fp)

        with open("X_train_database.txt", "rb") as fp:
            X_train = pickle.load(fp)

        with open("y_train_database.txt", "rb") as fp:
            y_train = pickle.load(fp)

        with open("X_test_database.txt", "rb") as fp:
            X_test = pickle.load(fp)

        with open("y_test_database.txt", "rb") as fp:
            y_test = pickle.load(fp)

        X_train, y_train = make_classification()
        DistanceMetric.get_metric('mahalanobis', V=np.cov(X_train))
        model_nn = KNeighborsClassifier(algorithm='brute',
                                        metric='mahalanobis',
                                        metric_params={'V': np.cov(X_train)})

        model_nn = KNeighborsClassifier(n_neighbors=3)

        model_nn.fit(X_train, y_train)
        predictions = model_nn.predict(X_test)
        print(predictions)
        knnPickle = open('model_knn', 'wb')
        pickle.dump(model_nn, knnPickle)

        # Declare the model
        # model = Sequential()

        # # Declare the layers
        # layer_1 = Conv2D(32, kernel_size=3, activation='relu',
        #                  input_shape=(28, 28, 1))
        # layer_2 = Conv2D(64, kernel_size=3, activation='relu')
        # layer_3 = Flatten()
        # layer_4 = Dense(10, activation='softmax')

        # # Add the layers to the model
        # model.add(layer_1)
        # model.add(layer_2)
        # model.add(layer_3)
        # model.add(layer_4)

        # model.compile(optimizer='adam', loss='categorical_crossentropy',
        #               metrics=['accuracy'])

        # model.fit(np.array(X_train), np.array(y_train), validation_data=(
        #     np.array(X_test), np.array(y_test)), epochs=10)

        # model.save("my_model")

    def test(processedDigits):
        model = pickle.load(open('model_knn', 'rb'))
        with open("X_test_database.txt", "rb") as fp:
            X_test = pickle.load(fp)
        inp = np.array(processedDigits)
        output = []
        result = model.predict(X_test)
        # result = model.predict(inp)
        for digit in inp:
            digit = digit.reshape(28, 28)
            # result = model.predict(digit)
            # output.append(result)
            print("\n\n---------------------------------------\n\n")
            print("=========PREDICTION============ \n\n")
            # plt.imshow(digit.reshape(28, 28), cmap="gray")
            # plt.show()

            # output.append(format(np.argmax(prediction)))

            # print(
            #     "\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
            # hard_maxed_prediction = np.zeros(prediction.shape)
            # hard_maxed_prediction[0][np.argmax(prediction)] = 1
            # print(
            #     "\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
            print("\n\n---------------------------------------\n\n")

        print(output)
