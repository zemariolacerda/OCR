import cv2
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn import metrics
import pickle
import warnings
from sklearn import svm
from scipy.ndimage import interpolation
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
    global testSVM
    global setDatabase
    global trainSVM

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

            print(a)
            print(b)
            a = np.array(a)
            b = np.array(b)

            array_sum = a + b
            i = 28
            z = i/len(array_sum)

            array_compress = interpolation.zoom(array_sum, z)

            projectedDigits.append(array_compress)

            print(array_compress)
            plt.plot(array_compress)
            plt.show()

        # projectedDigits.reshape(28, 28)
        return projectedDigits

    def projection_database(digit):
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

        a = np.array(a)
        b = np.array(b)

        array_sum = a + b
        i = 28
        z = i/len(array_sum)

        array_compress = interpolation.zoom(array_sum, z)

        return array_compress

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

    def setDatabase():
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # So, let’s reshape the dataset according to our model.
        # 60000 -> number of images, 28x28 -> size of each image, 1 -> image in greyScale
        # X_train = X_train.reshape(28, 28)
        # X_test = X_test.reshape(28, 28)
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)

        X_train_proj = []
        X_train_proj = Parallel(n_jobs=4)(
            delayed(process)(i) for i in X_train)

        X_test_proj = []
        X_test_proj = Parallel(n_jobs=4)(
            delayed(process)(i) for i in X_test)

        X_train = np.array(X_train_proj)
        X_test = np.array(X_test_proj)

        with open("X_train_database.txt", "wb") as fp:
            pickle.dump(X_train, fp)

        with open("y_train_database.txt", "wb") as fp:
            pickle.dump(y_train, fp)

        with open("X_test_database.txt", "wb") as fp:
            pickle.dump(X_test, fp)

        with open("y_test_database.txt", "wb") as fp:
            pickle.dump(y_test, fp)

    def trainSVM(digits):
        setDatabase()

        with open("X_train_database.txt", "rb") as fp:
            X_train = pickle.load(fp)

        with open("y_train_database.txt", "rb") as fp:
            y_train = pickle.load(fp)

        with open("X_test_database.txt", "rb") as fp:
            X_test = pickle.load(fp)

        with open("y_test_database.txt", "rb") as fp:
            y_test = pickle.load(fp)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(np.shape(X_train))
        print(np.shape(y_train))

        # print(X_train.shape())

        # (X_train, y_train), (X_test, y_test) = mnist.load_data()

        x_size = 60000
        y_size = 60000

        X_train = X_train[:x_size]
        y_train = y_train[:y_size]

        X_train = np.array(X_train).reshape(x_size, 28)
        X_test = np.array(X_test).reshape(10000, 28)

        print(np.shape(X_train))
        print(np.shape(y_train))

        clf = svm.SVC(kernel='rbf', verbose=True)

        clf.fit(X_train, y_train)

        filename = 'svm.sav'
        pickle.dump(clf, open(filename, 'wb'))

        y_pred = clf.predict(X_test)

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        # X_train = np.concatenate((X_train, y_train), axis=0)
        # X_test = np.concatenate((X_test, y_test), axis=0)

        # svm = LinearSVC(dual=False)
        # svm.fit(X_train, y_train)

        # svm.coef_
        # svm.intercept_

        # pred = svm.predict(X_test)
        # accuracy_score(y_test, pred)

        # cm = confusion_matrix(y_test, pred)

        # matplot.subplots(figsize=(10, 6))
        # sb.heatmap(cm, annot=True, fmt='g')
        # matplot.xlabel("Predicted")
        # matplot.ylabel("Actual")
        # matplot.title("Confusion Matrix")
        # matplot.show()

    def testSVM(digits):
        clf = pickle.load(open('svm.sav', 'rb'))

        digits = np.array(digits)
        total, size = np.shape(digits)
        # print(np.shape(digits))

        digits = np.array(digits).reshape(total, 28)
        y_pred = clf.predict(digits)

        print("Numeros:", y_pred)
        print("")
