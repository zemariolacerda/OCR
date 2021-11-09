# PONTIFICIA UNIVERSIDADE CATOLICA DE MINAS GERAIS
#
# PROCESSAMENTO DE IMAGENS
#
# José Mário de Carvalho Lacerda
# Pedro Lages Ribeiro
# Lucas Soriano de Oliveira

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import Label, filedialog
import Preprocessing as pr
import timeit

# Imagem de abertura do programa
fileName = "sequencia.png"
image_threshold = 0
original = 0
img_numbers = []
digits = []

# Canvas Loop
window = tk.Tk()

# Tamanho do Canvas
canvas = tk.Canvas(window, width=1000, height=1000)

# Botao procurar Imagem
browse_text = tk.StringVar()
browse_btn = tk.Button(window, textvariable=browse_text,
                       command=lambda: load(), font="Raleway")
browse_text.set("Procurar Imagem")
browse_btn.grid(column=0, row=0)

# Botao Threshold
threshold_text = tk.StringVar()
threshold_btn = tk.Button(window, textvariable=threshold_text,
                          command=lambda: threshold_img(), font="Raleway")
threshold_text.set("Threshold")
threshold_btn.grid(column=1, row=0)

# Botao Threshold invertido
inv_threshold_text = tk.StringVar()
inv_threshold_btn = tk.Button(window, textvariable=inv_threshold_text,
                              command=lambda: inverse_threshold_img(), font="Raleway")
inv_threshold_text.set("Inv. Threshold")
inv_threshold_btn.grid(column=2, row=0)

# Botao contornos
contour_img_text = tk.StringVar()
contour_img_btn = tk.Button(window, textvariable=contour_img_text,
                            command=lambda: select_contours(), font="Raleway")
contour_img_text.set("Contornar")
contour_img_btn.grid(column=3, row=0)

# Botao Projetar
projection_text = tk.StringVar()
projection_btn = tk.Button(window, textvariable=projection_text,
                           command=lambda: projection(), font="Raleway")
projection_text.set("Projetar")
projection_btn.grid(column=4, row=0)

# Botao Treinar SVM
train_text = tk.StringVar()
train_btn = tk.Button(window, textvariable=train_text,
                      command=lambda: trainSVM(), font="Raleway")
train_text.set("Treinar SVM")
train_btn.grid(column=5, row=0)

# Botao testar SVM
test_text = tk.StringVar()
test_btn = tk.Button(window, textvariable=test_text,
                     command=lambda: testSVM(), font="Raleway")
test_text.set("Testar SVM")
test_btn.grid(column=6, row=0)

# Botao treinar Sequencial
train_text = tk.StringVar()
train_btn = tk.Button(window, textvariable=train_text,
                      command=lambda: trainSeq(), font="Raleway")
train_text.set("Treinar Seq")
train_btn.grid(column=7, row=0)

# Botao testar Sequencial
test_text = tk.StringVar()
test_btn = tk.Button(window, textvariable=test_text,
                     command=lambda: testSeq(), font="Raleway")
test_text.set("Testar Seq")
test_btn.grid(column=8, row=0)

# Abre a imagem inicial do programa
initial_image = ImageTk.PhotoImage(Image.open(fileName))
image_label = Label(image=initial_image)
image_label.grid(column=0, row=1, columnspan=9)


# Treina SVM
def trainSVM():
    global digits
    t = timeit.timeit(lambda: pr.trainSVM(digits), number=1)
    print("Tempo de execucao: ", t, "s.")


# Treina Sequencial
def trainSeq():
    global digits
    t = timeit.timeit(lambda: pr.trainSeq(digits), number=1)
    print("Tempo de execucao: ", t, "s.")


# Seleciona os digitos da imagem e mostra imagem com contornos
def select_contours():
    global fileName
    global image_threshold
    global img_numbers
    img_numbers = pr.selectDigits(fileName, image_threshold)


# Aplica Threshold na imagem
def threshold_img():
    global fileName
    global image_label
    global image_threshold
    global original
    image_threshold, image_array = pr.threshold(fileName)
    image_label.grid_forget()
    image = ImageTk.PhotoImage(image=Image.fromarray(image_array[2]))
    image_label = tk.Label(image=image)
    image_label.image = image
    image_label.grid(column=0, row=1, columnspan=9)


# Aplica Threshold invertido na imagem
def inverse_threshold_img():
    global fileName
    global image_label
    global image_threshold
    image_threshold, image_array = pr.inv_threshold(fileName)
    image_label.grid_forget()
    image = ImageTk.PhotoImage(image=Image.fromarray(image_array[2]))
    image_label = tk.Label(image=image)
    image_label.image = image
    image_label.grid(column=0, row=1, columnspan=9)


# Metodo de gerenciamento de arquivos para selecionar uma nova imagem
def browseFiles():
    global fileName
    fileName = filedialog.askopenfile(title="Select a File",
                                      filetypes=(("PNG",
                                                  "*.png*"),
                                                 ("JPG",
                                                 "*.jpg*")))
    return fileName.name


# Carrega nova imagem
def load():
    global fileName
    global image_label
    image_label.grid_forget()
    fileName = browseFiles()
    image = Image.open(fileName)
    image = ImageTk.PhotoImage(image)
    image_label = tk.Label(image=image)
    image_label.image = image
    image_label.grid(column=0, row=1, columnspan=9)


# Calcula as projecoes
def projection():
    global digits
    digits = pr.projection(img_numbers)


# Testa SVM
def testSVM():
    global digits
    pr.testSVM(digits)


# Testa Sequencial
def testSeq():
    global digits
    pr.testSeq(digits)


# Fim do loop do Canvas
window.mainloop()
