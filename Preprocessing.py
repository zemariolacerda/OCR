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
import tensorflow as tf
from tensorflow import keras

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
    global define_model
    global trainRNN
    global testRNN

    def sort_contours(cnts):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i]))

        return cnts

    def selectDigits(img_directory, thresh):
        # Abre imagem original
        image = cv2.imread(img_directory)

        # Encontra os contornos da imagem
        _, contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ordena os contornos da esquerda para a direita
        contours = sort_contours(contours)

        preprocessed_digits = []

        for c in contours:
            # Define as posicoes na imagem
            x, y, w, h = cv2.boundingRect(c)

            # Cria um retangulo verde na imagem original na posicao do contorno
            cv2.rectangle(image, (x, y), (x+w, y+h),
                          color=(0, 255, 0), thickness=2)

            # Recorta o digito da imagem binarizada
            digit = thresh[y:y+h, x:x+w]

            # Padroniza os contornos para que eles fiquem quadrados
            digit_w, digit_h = digit.shape
            value = abs(digit_h - digit_w)
            digit = np.pad(digit, ((0, 0), (value//2, value//2)),
                           "constant", constant_values=0)

            # Altera o tamanho do digito para 18x18
            resized_digit = cv2.resize(digit, (18, 18))

            # Adiciona padding de 5 em todas as direcoes e preenche com pixels pretos
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)),
                                  "constant", constant_values=0)

            # Adiciona digito a lista de digitos ja processados
            preprocessed_digits.append(padded_digit)

        # Mostra imagem original (RGB) com contornos
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_img)
        plt.show()

        return np.array(preprocessed_digits)

    def threshold(img_dir):

        # Carrega imagem com tons de cinza (parametro 0)
        image = cv2.imread(img_dir, 0)

        # Aplica blur
        blur = cv2.GaussianBlur(image, (5, 5), 0)

        # Aplica Treshold
        ret, threshold = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        image_array = [blur, 0, threshold]

        return threshold, image_array

    def threshold_database(image):
        # Recebe imagem da base e aplica blur
        blur = cv2.GaussianBlur(image, (5, 5), 0)

        # Aplica Threshold
        ret, threshold = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def inv_threshold(img_dir):
        # Carrega imagem com tons de cinza (parametro 0)
        image = cv2.imread(img_dir, 0)

        # Aplica blur
        blur = cv2.GaussianBlur(image, (5, 5), 0)

        # Aplica Treshold invertido
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

            for j in range(0, w):  # Caminha pela coluna
                for i in range(0, h):  # Caminha pela linha
                    if vertical_p[i, j] == 0:  # Verifica se o ponto é preto
                        a[j] += 1  # Contador vertical
                        vertical_p[i, j] = 255  # Transforma o ponto em branco

            for j in range(0, w):  # Caminha pelas colunas
                # Comeca da posicao mais alta da coluna que deveria ser preta ate o fim
                for i in range((h-a[j]), h):
                    vertical_p[i, j] = 0  # Transforma o ponto em preto

            for j in range(0, h):  # Caminha pela linha
                for i in range(0, w):  # Caminha pela coluna
                    if horizontal_p[j, i] == 0:  # Verifica se o ponto é preto
                        b[j] += 1   # Contador horizontal
                        # Transforma o ponto em branco
                        horizontal_p[j, i] = 255

            for j in range(0, h):  # Caminha pela linha
                # Comeca da posicao mais alta da linha que deveria ser preta ate o fim
                for i in range(0, b[j]):
                    horizontal_p[j, i] = 0  # Transforma ponto em preto

            # Projecao concatenada formada pela uniao da projecao vertical e horizontal
            concat_p = vertical_p + horizontal_p

            # Plotagem dos graficos
            fig = plt.figure(figsize=(10, 7))

            # Mostra o digito recortado
            fig.add_subplot(1, 4, 1)
            plt.imshow(digit.reshape(28, 28), cmap="gray")
            plt.axis('off')
            plt.title("Digito")

            # Mostra a projecao vertical
            fig.add_subplot(1, 4, 2)
            plt.imshow(vertical_p, cmap=plt.gray())
            plt.title("Projecao Vertical")

            # Mostra a projecao horizontal
            fig.add_subplot(1, 4, 3)
            plt.imshow(horizontal_p, cmap=plt.gray())
            plt.title("Projecao Horizontal")

            # Mostra a projecao concatenada
            fig.add_subplot(1, 4, 4)
            plt.imshow(concat_p, cmap=plt.gray())
            plt.title("Projecao Concatenada")
            plt.show()

            # interpopla a soma dos arrays (originalmente um array de 56 posicoes) num array de tamanho 28
            a = np.array(a)
            b = np.array(b)
            array_sum = a + b
            i = 28
            z = i/len(array_sum)
            array_compress = interpolation.zoom(array_sum, z)

            projectedDigits.append(array_compress)

        return np.array(projectedDigits)

    def projection_database(digit):
        vertical_p = digit.copy()
        horizontal_p = digit.copy()
        concat_p = digit.copy()
        (h, w) = vertical_p.shape
        (h, w) = horizontal_p.shape
        a = [0 for z in range(0, w)]
        b = [0 for z in range(0, w)]

        for j in range(0, w):  # Caminha pela coluna
            for i in range(0, h):  # Caminha pela linha
                if vertical_p[i, j] == 0:  # Verifica se o ponto é preto
                    a[j] += 1  # Contador vertical
                    vertical_p[i, j] = 255  # Transforma o ponto em branco

        for j in range(0, w):  # Caminha pelas colunas
            # Comeca da posicao mais alta da coluna que deveria ser preta ate o fim
            for i in range((h-a[j]), h):
                vertical_p[i, j] = 0  # Transforma o ponto em preto

        for j in range(0, h):  # Caminha pela linha
            for i in range(0, w):  # Caminha pela coluna
                if horizontal_p[j, i] == 0:  # Verifica se o ponto é preto
                    b[j] += 1   # Contador horizontal
                    # Transforma o ponto em branco
                    horizontal_p[j, i] = 255

        for j in range(0, h):  # Caminha pela linha
            # Comeca da posicao mais alta da linha que deveria ser preta ate o fim
            for i in range(0, b[j]):
                horizontal_p[j, i] = 0  # Transforma ponto em preto

        # interpopla a soma dos arrays (originalmente um array de 56 posicoes) num array de tamanho 28
        a = np.array(a)
        b = np.array(b)
        array_sum = a + b
        i = 28
        z = i/len(array_sum)
        array_compress = interpolation.zoom(array_sum, z)

        return array_compress

    # Retorna a projecao interpolada dos numeros da base MNIST
    def process(i):
        return projection_database(i)

    # Processa as imagens da base para prepara-las para o aprendizado de maquina
    def setDatabase():
        # Carrega a base do mnist. x_train
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        X_train_proj = []
        X_train_proj = Parallel(n_jobs=4)(
            delayed(process)(i) for i in x_train)

        X_test_proj = []
        X_test_proj = Parallel(n_jobs=4)(
            delayed(process)(i) for i in x_test)

        x_train = np.array(X_train_proj)
        x_test = np.array(X_test_proj)

        with open("X_train_database.txt", "wb") as fp:
            pickle.dump(x_train, fp)

        with open("y_train_database.txt", "wb") as fp:
            pickle.dump(y_train, fp)

        with open("X_test_database.txt", "wb") as fp:
            pickle.dump(x_test, fp)

        with open("y_test_database.txt", "wb") as fp:
            pickle.dump(y_test, fp)

    def trainSVM(digits):
        # setDatabase()

        with open("X_train_database.txt", "rb") as fp:
            x_train = pickle.load(fp)

        with open("y_train_database.txt", "rb") as fp:
            y_train = pickle.load(fp)

        with open("X_test_database.txt", "rb") as fp:
            x_test = pickle.load(fp)

        with open("y_test_database.txt", "rb") as fp:
            y_test = pickle.load(fp)

        x_size = 10000
        y_size = 10000

        x_train = x_train[:x_size]
        y_train = y_train[:y_size]

        x_train = np.array(x_train).reshape(x_size, 28)
        x_test = np.array(x_test).reshape(10000, 28)

        svmModel = svm.SVC(kernel='poly', degree=5, verbose=True)

        svmModel.fit(x_train, y_train)

        filename = 'svm.sav'
        pickle.dump(svmModel, open(filename, 'wb'))

        y_pred = svmModel.predict(x_test)

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    def testSVM(digits):
        clf = pickle.load(open('svm.sav', 'rb'))

        digits = np.array(digits)
        total, size = np.shape(digits)
        # print(np.shape(digits))

        digits = np.array(digits).reshape(total, 28)
        y_pred = clf.predict(digits)

        print("Numeros:", y_pred)

    def trainRNN(digits):
        setDatabase()

        with open("X_train_database.txt", "rb") as fp:
            x_train = pickle.load(fp)

        with open("y_train_database.txt", "rb") as fp:
            y_train = pickle.load(fp)

        with open("X_test_database.txt", "rb") as fp:
            x_test = pickle.load(fp)

        with open("y_test_database.txt", "rb") as fp:
            y_test = pickle.load(fp)

        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(28, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, validation_data=(
            x_test, y_test), epochs=10)
        model.save('resources/NNN')

    def testRNN(digits):
        with open("X_test_database.txt", "rb") as fp:
            x_test = pickle.load(fp)

        with open("y_test_database.txt", "rb") as fp:
            y_test = pickle.load(fp)

        model = tf.keras.models.load_model('resources/NNN')

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print("Precisao: ", test_acc)

        y_pred = []
        for d in digits:
            pred = model.predict(d.reshape(1, 28))
            print(np.argmax(pred), end=" ")
            y_pred.append(pred)
