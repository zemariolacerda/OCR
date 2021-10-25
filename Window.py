import tkinter as tk
from PIL import ImageTk, Image
from tkinter import Label, filedialog
import Preprocessing as pr
import timeit
# import Preprocessing

fileName = "imagem.png"
image_threshold = 0
original = 0
img_numbers = []
digits = []

# Canvas Loop
window = tk.Tk()

canvas = tk.Canvas(window, width=1000, height=1000)
# canvas.grid(columnspan=10, rowspan=10)
# canvas.rowconfigure(0, weight=9)
# canvas.columnconfigure(0, weight=9)

browse_text = tk.StringVar()
browse_btn = tk.Button(window, textvariable=browse_text,
                       command=lambda: load(), font="Raleway")
browse_text.set("Procurar Imagem")
browse_btn.grid(column=0, row=0)

threshold_text = tk.StringVar()
threshold_btn = tk.Button(window, textvariable=threshold_text,
                          command=lambda: threshold_img(), font="Raleway")
threshold_text.set("Threshold")
threshold_btn.grid(column=1, row=0)

inv_threshold_text = tk.StringVar()
inv_threshold_btn = tk.Button(window, textvariable=inv_threshold_text,
                              command=lambda: inverse_threshold_img(), font="Raleway")
inv_threshold_text.set("Inv. Threshold")
inv_threshold_btn.grid(column=2, row=0)

contour_img_text = tk.StringVar()
contour_img_btn = tk.Button(window, textvariable=contour_img_text,
                            command=lambda: select_contours(), font="Raleway")
contour_img_text.set("Contornar")
contour_img_btn.grid(column=3, row=0)

projection_text = tk.StringVar()
projection_btn = tk.Button(window, textvariable=projection_text,
                           command=lambda: projection(), font="Raleway")
projection_text.set("Projetar")
projection_btn.grid(column=4, row=0)

train_text = tk.StringVar()
train_btn = tk.Button(window, textvariable=train_text,
                      command=lambda: train(), font="Raleway")
train_text.set("Treinar")
train_btn.grid(column=5, row=0)

test_text = tk.StringVar()
test_btn = tk.Button(window, textvariable=test_text,
                     command=lambda: test(), font="Raleway")
test_text.set("Testar")
test_btn.grid(column=6, row=0)


initial_image = ImageTk.PhotoImage(Image.open(fileName))
image_label = Label(image=initial_image)
image_label.grid(column=0, row=1, columnspan=7)


def train():
    global digits
    t = timeit.timeit(lambda: pr.trainSVM(digits), number=1)
    print(t)


def select_contours():
    global fileName
    global image_threshold
    global img_numbers
    img_numbers = pr.selectDigits(fileName, image_threshold)


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
    image_label.grid(column=0, row=1, columnspan=7)


def inverse_threshold_img():
    global fileName
    global image_label
    global image_threshold
    image_threshold, image_array = pr.inv_threshold(fileName)
    image_label.grid_forget()
    image = ImageTk.PhotoImage(image=Image.fromarray(image_array[2]))
    image_label = tk.Label(image=image)
    image_label.image = image
    image_label.grid(column=0, row=1, columnspan=7)


def browseFiles():
    global fileName
    fileName = filedialog.askopenfile(title="Select a File",
                                      filetypes=(("PNG",
                                                  "*.png*"),
                                                 ("JPG",
                                                 "*.jpg*")))
    return fileName.name


def load():
    global fileName
    global image_label
    image_label.grid_forget()
    fileName = browseFiles()
    image = Image.open(fileName)
    image = ImageTk.PhotoImage(image)
    image_label = tk.Label(image=image)
    image_label.image = image
    image_label.grid(column=0, row=1, columnspan=7)


def projection():
    global digits
    digits = pr.projection(img_numbers)


def test():
    global digits
    t = timeit.timeit(lambda: pr.testSVM(digits), number=1)
    print(t)


window.mainloop()
# Create Canvas
