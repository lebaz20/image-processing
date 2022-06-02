

from ast import Try
from tkinter import *
from skimage import data
from skimage import color
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from scipy import ndimage
import numpy as np
from skimage import data, io
from skimage.io import imread
from matplotlib.pyplot import imshow, show, subplot, title, get_cmap
from skimage.color import rgb2gray, gray2rgb, rgb2hsv, rgb2lab, lab2rgb
from skimage import exposure
from skimage.exposure import equalize_hist
from skimage.util import random_noise
from io import BytesIO
# loading Python Imaging Library
from PIL import ImageTk, Image, ImageOps
# To get the dialog box to open when required
from tkinter import filedialog
import cv2

def reset_plt():
    plt.figure().clear()

def to_pil(image: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # return Image.fromarray(image)

    # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that 
    # the color is converted from BGR to RGB
    # color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pil_image=Image.fromarray(color_coverted)
    # return pil_image

    return Image.fromarray((image * 1).astype(np.uint8)).convert('RGB')

def to_cv2(image: Image) -> np.ndarray:
    # return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    # return np.asarray(image)

    # use numpy to convert the pil_image into a numpy array
    numpy_image=np.array(image)  

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
    # the color is converted from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image

def openfilename():
 
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title ='"pen')
    return filename

def resize_img(image):
    # resize the image and apply a high-quality down sampling filter
    width, height = image.size
    if height > 400:
        new_height = 300
        ratio = width / height
        new_width = int(ratio * new_height)
    else :
        ratio = height / width
        new_width = 500
        new_height = int(ratio * new_width)
    resizedImg = image.resize((new_width, new_height), Image.ANTIALIAS) 
    return resizedImg

def open_img():
    # Select the Imagename  from a folder
    global filename
    filename = openfilename()
 
    # opens the image
    global img
    img = Image.open(filename)

    resized_img = resize_img(img)
 
    # PhotoImage class is used to add image to widgets, icons etc
    tk_img = ImageTk.PhotoImage(resized_img)
  
    global panel1
    # create a label
    panel1 = Label(root, image = tk_img)
     
    # set the image as img
    panel1.image = tk_img
    panel1.grid(row = 1, rowspan = 13, column = 1, padx=10)

def original_histogram():
    img_grey = ImageOps.grayscale(img)
    plt.hist(np.asarray(img_grey).ravel(),256,[0,256])

    img_data = BytesIO()
    plt.savefig(img_data)
    reset_plt()
    load = Image.open(img_data)
    load = resize_img(load)
    image_from_plot = ImageTk.PhotoImage(load)

    global panel2
    # create a label
    panel2 = Label(root, image = image_from_plot)
    # set the image as img
    panel2.image = image_from_plot
    panel2.grid(row = 1, rowspan = 13, column = 2, padx=10)

def equalized_histogram():
    img_equalized = ImageOps.equalize(img.convert('RGB'), mask = None)

    plt.hist(np.asarray(ImageOps.grayscale(img_equalized)).ravel(),256,[0,256])
   
    img_equalized = resize_img(img_equalized)
    img_equalized = ImageTk.PhotoImage(img_equalized)
    global panel3
    # create a label
    panel3 = Label(root, image = img_equalized) 
    # set the image as img
    panel3.image = img_equalized
    panel3.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

    img_data = BytesIO()
    plt.savefig(img_data)
    reset_plt()
    load = Image.open(img_data)
    load = resize_img(load)
    image_from_plot = ImageTk.PhotoImage(load)
    global panel4
    # create a label
    panel4 = Label(root, image = image_from_plot)
    # set the image as img
    panel4.image = image_from_plot
    panel4.grid(row = 14, rowspan = 13, column = 2, padx=10, pady=10)

def init_sobel():
    global sobelX
    sobelX = IntVar()
    global sobelXCheck
    sobelXCheck = Checkbutton(root, text='X', variable=sobelX, onvalue=1, offvalue=0)
    sobelXCheck.grid(row = 21, column = 0, sticky="nsw")

    global sobelY
    sobelY = IntVar()
    global sobelYCheck
    sobelYCheck = Checkbutton(root, text='Y', variable=sobelY, onvalue=1, offvalue=0)
    sobelYCheck.grid(row = 22, column = 0, sticky="nsw")

    global sobelKLabel
    sobelKLabel = Label(root, text='Kernel Size:')
    sobelKLabel.grid(row = 23, column = 0, sticky="nsw")
    global sobelK
    sobelK = StringVar()
    global sobelKEntry
    sobelKEntry = Entry(root, textvariable=sobelK)
    sobelKEntry.grid(row = 24, column = 0, sticky="nesw")

    global sobelSubmit
    sobelSubmit = Button(root, text =' Submit', anchor="w", command = sobel)
    sobelSubmit.grid(row = 25, column = 0, sticky="nesw")

def sobel():
    img_gray_cv = cv2.imread(filename, flags=0)

    # remove noise
    img_blur = cv2.GaussianBlur(img_gray_cv,(3,3), sigmaX=0, sigmaY=0)

    cv2_sobel = cv2.Sobel(src=img_blur, ddepth=cv2.CV_32F, dx=sobelX.get(), dy=sobelY.get(), ksize=int(sobelK.get()))
    img_sobel = resize_img(to_pil(cv2_sobel))
    img_sobel = ImageTk.PhotoImage(img_sobel)
    # sobel_img = convert_from_cv2_to_image(sobel_cv2)

    global sobelPanel
    # create a label
    sobelPanel = Label(root, image = img_sobel) 
    # set the image as img
    sobelPanel.image = img_sobel
    sobelPanel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

def reset_sobel():
    try:
        sobelXCheck.grid_remove()
        print("sobelXCheck removed")
    except:
        print("sobelXCheck undefined")
    try:
        sobelYCheck.grid_remove()
        print("sobelYCheck removed")
    except:
        print("sobelYCheck undefined")
    try:
        sobelKLabel.grid_remove()
        print("sobelKLabel removed")
    except:
        print("sobelKLabel undefined")
    try:
        sobelKEntry.grid_remove()
        print("sobelKEntry removed")
    except:
        print("sobelKEntry undefined")
    try:
        sobelSubmit.grid_remove()
        print("sobelSubmit removed")
    except:
        print("sobelSubmit undefined")
    try:
        sobelPanel.grid_remove()
        print("sobelPanel removed")
    except:
        print("sobelPanel undefined")

def init_laplace():
    global laplaceKLabel
    laplaceKLabel = Label(root, text='Kernel Size:')
    laplaceKLabel.grid(row = 23, column = 0, sticky="nsw")
    global laplaceK
    laplaceK = StringVar()
    global laplaceKEntry
    laplaceKEntry = Entry(root, textvariable=laplaceK)
    laplaceKEntry.grid(row = 24, column = 0, sticky="nesw")

    global laplaceSubmit
    laplaceSubmit = Button(root, text =' Submit', anchor="w", command = laplace)
    laplaceSubmit.grid(row = 25, column = 0, sticky="nesw")

def laplace():
    img_gray_cv = cv2.imread(filename, flags=0)

    # remove noise
    img_blur = cv2.GaussianBlur(img_gray_cv,(3,3), sigmaX=0, sigmaY=0)

    cv2_laplace = cv2.Laplacian(src=img_blur, ddepth=cv2.CV_32F, ksize=int(laplaceK.get()))
    img_laplace = resize_img(to_pil(cv2_laplace))
    img_laplace = ImageTk.PhotoImage(img_laplace)

    global laplacePanel
    # create a label
    laplacePanel = Label(root, image = img_laplace) 
    # set the image as img
    laplacePanel.image = img_laplace
    laplacePanel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

def reset_laplace():
    try:
        laplaceKLabel.grid_remove()
        print("laplaceKLabel removed")
    except:
        print("laplaceKLabel undefined")
    try:
        laplaceKEntry.grid_remove()
        print("laplaceKEntry removed")
    except:
        print("laplaceKEntry undefined")
    try:
        laplaceSubmit.grid_remove()
        print("laplaceSubmit removed")
    except:
        print("laplaceSubmit undefined")
    try:
        laplacePanel.grid_remove()
        print("laplacePanel removed")
    except:
        print("laplacePanel undefined")

def fourier_transform():
    img_gray_cv = cv2.imread(filename, flags=0)

    f = np.fft.fft2(img_gray_cv)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    plt.imshow(magnitude_spectrum, cmap = 'gray')
    img_data = BytesIO()
    plt.savefig(img_data)
    reset_plt()
    load = Image.open(img_data)
    load = resize_img(load)
    image_from_plot = ImageTk.PhotoImage(load)

    global fourierPanel
    # create a label
    fourierPanel = Label(root, image = image_from_plot) 
    # set the image as img
    fourierPanel.image = image_from_plot
    fourierPanel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

def reset_fourier():
    try:
        fourierPanel.grid_remove()
        print("fourierPanel removed")
    except:
        print("fourierPanel undefined")

def reset():
    try:
        panel1.grid_remove()
        print("panel1 removed")
    except:
        print("panel1 undefined")
    try:
        panel2.grid_remove()
        print("panel2 removed")
    except:
        print("panel2 undefined")
    try:
        panel3.grid_remove()
        print("panel3 removed")
    except:
        print("panel3 undefined")
    try:
        panel4.grid_remove()
        print("panel4 removed")
    except:
        print("panel4 undefined")
    
    reset_sobel()
    reset_laplace()
    reset_fourier()


# Create a window
root = Tk()
root.columnconfigure(0, minsize=300)

# Set Title as Image Loader
root.title("Image Processing")
 
# Set the resolution of window
root.geometry("1400x900+300+150")
 
# Allow Window to be resizable
root.resizable(width = True, height = True)
 
sepFrameHeight = 20

# Create a button and place it into the window using grid layout
btn = Button(root, text =' Open Image', anchor="w", command = open_img)
btn.grid(row = 1, sticky="nesw")

Frame(root, width=300, height=sepFrameHeight).grid(column=0, row = 2)

btn2 = Button(root, text =' Generate Original Histogram', anchor="w", command = original_histogram)
btn2.grid(row = 3, column = 0, sticky="nesw")

btn3 = Button(root, text =' Generate Equalized Histogram', anchor="w", command = equalized_histogram)
btn3.grid(row = 4, column = 0, sticky="nesw")

Frame(root, width=300, height=sepFrameHeight).grid(column=0, row = 5)

btn4 = Button(root, text =' Apply Sobel Filter', anchor="w", command = init_sobel)
btn4.grid(row = 6, column = 0, sticky="nesw")

btn5 = Button(root, text =' Apply Laplace Filter', anchor="w", command = init_laplace)
btn5.grid(row = 7, column = 0, sticky="nesw")

Frame(root, width=300, height=sepFrameHeight).grid(column=0, row = 8)

btn6 = Button(root, text =' Apply Fourier Transform', anchor="w", command = fourier_transform)
btn6.grid(row = 9, column = 0, sticky="nesw")

Frame(root, width=300, height=sepFrameHeight).grid(column=0, row = 10)

btn7 = Button(root, text =' Add Salt And Pepper Noise', anchor="w")
btn7.grid(row = 11, column = 0, sticky="nesw")

btn8 = Button(root, text =' Add Periodic Noise', anchor="w")
btn8.grid(row = 12, column = 0, sticky="nesw")

Frame(root, width=300, height=sepFrameHeight).grid(column=0, row = 13)

btn9 = Button(root, text =' Remove Salt And Pepper Noise By Median', anchor="w")
btn9.grid(row = 14, column = 0, sticky="nesw")

btn10 = Button(root, text =' Remove Periodic Noise By Mask', anchor="w")
btn10.grid(row = 15, column = 0, sticky="nesw")

btn11 = Button(root, text =' Remove Periodic Noise By Notch', anchor="w")
btn11.grid(row = 16, column = 0, sticky="nesw")

btn12 = Button(root, text =' Remove Periodic Noise By Band-Reject', anchor="w")
btn12.grid(row = 17, column = 0, sticky="nesw")

Frame(root, width=300, height=sepFrameHeight).grid(column=0, row = 18)

btn13 = Button(root, text =' Reset', anchor="w", command = reset)
btn13.grid(row = 19, column = 0, sticky="nesw")

Frame(root, width=300, height=sepFrameHeight).grid(column=0, row = 20)

root.mainloop()



