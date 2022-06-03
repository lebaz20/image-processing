

from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
from io import BytesIO
# loading Python Imaging Library
from PIL import ImageTk, Image, ImageOps
# To get the dialog box to open when required
from tkinter import filedialog
# OpenCV
import cv2

def reset_plt():
    plt.figure().clear()

def to_pil(image: np.ndarray) -> Image:
    # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that 
    # the color is converted from BGR to RGB
    return Image.fromarray((image * 1).astype(np.uint8)).convert('RGB')

def to_cv2(image: Image) -> np.ndarray:
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

def plt_to_img(bbox_inches=None, pad_inches=0.1):
    img_data = BytesIO()
    plt.savefig(img_data, bbox_inches=bbox_inches, pad_inches=pad_inches)
    reset_plt()
    load = Image.open(img_data)
    load = resize_img(load)
    return ImageTk.PhotoImage(load)

def open_img():
    global panel_1
    try:
        panel_1.grid_remove()
        print("panel_1 removed")
    except:
        print("panel_1 undefined")

    # Select the Imagename  from a folder
    global filename
    filename = openfilename()
 
    # opens the image
    global img
    img = Image.open(filename)

    # resize image
    resized_img = resize_img(img)
 
    # PhotoImage class is used to add image to widgets, icons etc
    tk_img = ImageTk.PhotoImage(resized_img)
  
    # create a label
    panel_1 = Label(root, image = tk_img)
     
    # set the image as img
    panel_1.image = tk_img
    panel_1.grid(row = 1, rowspan = 13, column = 1, padx=10)

def original_histogram():
    global original_hist_panel
    try:
        original_hist_panel.grid_remove()
        print("original_hist_panel removed")
    except:
        print("original_hist_panel undefined")

    img_grey = ImageOps.grayscale(img)
    plt.hist(np.asarray(img_grey).ravel(),256,[0,256])

    image_from_plot = plt_to_img()

    # create a label
    original_hist_panel = Label(root, image = image_from_plot)
    # set the image as img
    original_hist_panel.image = image_from_plot
    original_hist_panel.grid(row = 1, rowspan = 13, column = 2, padx=10)

def reset_original_histogram():
    try:
        original_hist_panel.grid_remove()
        print("original_hist_panel removed")
    except:
        print("original_hist_panel undefined")

def equalized_histogram():
    global equalized_img_panel
    try:
        equalized_img_panel.grid_remove()
        print("equalized_img_panel removed")
    except:
        print("equalized_img_panel undefined")
    global equalized_hist_panel
    try:
        equalized_hist_panel.grid_remove()
        print("equalized_hist_panel removed")
    except:
        print("equalized_hist_panel undefined")

    img_equalized = ImageOps.equalize(img.convert('RGB'), mask = None)
    plt.hist(np.asarray(ImageOps.grayscale(img_equalized)).ravel(),256,[0,256])
   
    img_equalized = resize_img(img_equalized)
    img_equalized = ImageTk.PhotoImage(img_equalized)

    # create a label
    equalized_img_panel = Label(root, image = img_equalized) 
    # set the image as img
    equalized_img_panel.image = img_equalized
    equalized_img_panel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

    image_from_plot = plt_to_img()

    # create a label
    equalized_hist_panel = Label(root, image = image_from_plot)
    # set the image as img
    equalized_hist_panel.image = image_from_plot
    equalized_hist_panel.grid(row = 14, rowspan = 13, column = 2, padx=10, pady=10)

def reset_equalized_histogram():
    try:
        equalized_img_panel.grid_remove()
        print("equalized_img_panel removed")
    except:
        print("equalized_img_panel undefined")
    try:
        equalized_hist_panel.grid_remove()
        print("equalized_hist_panel removed")
    except:
        print("equalized_hist_panel undefined")

def init_sobel():
    global sobel_x
    sobel_x = IntVar()
    global sobel_x_check
    sobel_x_check = Checkbutton(root, text='X', variable=sobel_x, onvalue=1, offvalue=0)
    sobel_x_check.grid(row = 21, column = 0, sticky="nsw")

    global sobel_y
    sobel_y = IntVar()
    global sobel_y_check
    sobel_y_check = Checkbutton(root, text='Y', variable=sobel_y, onvalue=1, offvalue=0)
    sobel_y_check.grid(row = 22, column = 0, sticky="nsw")

    global sobel_k_label
    sobel_k_label = Label(root, text='Kernel Size:')
    sobel_k_label.grid(row = 23, column = 0, sticky="nsw")
    global sobel_k
    sobel_k = StringVar()
    global sobel_k_entry
    sobel_k_entry = Entry(root, textvariable=sobel_k)
    sobel_k_entry.grid(row = 24, column = 0, sticky="nesw")

    global sobel_submit
    sobel_submit = Button(root, text =' Submit', anchor="w", command = sobel)
    sobel_submit.grid(row = 25, column = 0, sticky="nesw")

def sobel():
    global sobel_panel
    try:
        sobel_panel.grid_remove()
        print("sobel_panel removed")
    except:
        print("sobel_panel undefined")

    img_gray_cv = cv2.imread(filename, flags=0)

    # remove noise
    img_blur = cv2.GaussianBlur(img_gray_cv,(3,3), sigmaX=0, sigmaY=0)

    cv2_sobel = cv2.Sobel(src=img_blur, ddepth=cv2.CV_32F, dx=sobel_x.get(), dy=sobel_y.get(), ksize=int(sobel_k.get()))
    img_sobel = resize_img(to_pil(cv2_sobel))
    img_sobel = ImageTk.PhotoImage(img_sobel)

    # create a label
    sobel_panel = Label(root, image = img_sobel) 
    # set the image as img
    sobel_panel.image = img_sobel
    sobel_panel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

def reset_sobel():
    try:
        sobel_x_check.grid_remove()
        print("sobel_x_check removed")
    except:
        print("sobel_x_check undefined")
    try:
        sobel_y_check.grid_remove()
        print("sobel_y_check removed")
    except:
        print("sobel_y_check undefined")
    try:
        sobel_k_label.grid_remove()
        print("sobel_k_label removed")
    except:
        print("sobel_k_label undefined")
    try:
        sobel_k_entry.grid_remove()
        print("sobel_k_entry removed")
    except:
        print("sobel_k_entry undefined")
    try:
        sobel_submit.grid_remove()
        print("sobel_submit removed")
    except:
        print("sobel_submit undefined")
    try:
        sobel_panel.grid_remove()
        print("sobel_panel removed")
    except:
        print("sobel_panel undefined")

def init_laplace():
    global laplace_k_label
    laplace_k_label = Label(root, text='Kernel Size:')
    laplace_k_label.grid(row = 23, column = 0, sticky="nsw")
    global laplace_k
    laplace_k = StringVar()
    global laplace_k_entry
    laplace_k_entry = Entry(root, textvariable=laplace_k)
    laplace_k_entry.grid(row = 24, column = 0, sticky="nesw")

    global laplace_submit
    laplace_submit = Button(root, text =' Submit', anchor="w", command = laplace)
    laplace_submit.grid(row = 25, column = 0, sticky="nesw")

def laplace():
    global laplace_panel
    try:
        laplace_panel.grid_remove()
        print("laplace_panel removed")
    except:
        print("laplace_panel undefined")

    img_gray_cv = cv2.imread(filename, flags=0)

    # remove noise
    img_blur = cv2.GaussianBlur(img_gray_cv,(3,3), sigmaX=0, sigmaY=0)

    cv2_laplace = cv2.Laplacian(src=img_blur, ddepth=cv2.CV_32F, ksize=int(laplace_k.get()))
    img_laplace = resize_img(to_pil(cv2_laplace))
    img_laplace = ImageTk.PhotoImage(img_laplace)

    # create a label
    laplace_panel = Label(root, image = img_laplace) 
    # set the image as img
    laplace_panel.image = img_laplace
    laplace_panel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

def reset_laplace():
    try:
        laplace_k_label.grid_remove()
        print("laplace_k_label removed")
    except:
        print("laplace_k_label undefined")
    try:
        laplace_k_entry.grid_remove()
        print("laplace_k_entry removed")
    except:
        print("laplace_k_entry undefined")
    try:
        laplace_submit.grid_remove()
        print("laplace_submit removed")
    except:
        print("laplace_submit undefined")
    try:
        laplace_panel.grid_remove()
        print("laplace_panel removed")
    except:
        print("laplace_panel undefined")

def fourier_transform():
    global fourier_panel
    try:
        fourier_panel.grid_remove()
        print("fourier_panel removed")
    except:
        print("fourier_panel undefined")

    img_gray_cv = cv2.imread(filename, flags=0)

    f = np.fft.fft2(img_gray_cv)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    plt.imshow(magnitude_spectrum, cmap = 'gray')
    image_from_plot = plt_to_img()

    # create a label
    fourier_panel = Label(root, image = image_from_plot) 
    # set the image as img
    fourier_panel.image = image_from_plot
    fourier_panel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

def reset_fourier():
    try:
        fourier_panel.grid_remove()
        print("fourier_panel removed")
    except:
        print("fourier_panel undefined")

def init_salt_pepper():
    reset_median()

    global salt_a_label
    salt_a_label = Label(root, text='Amount:')
    salt_a_label.grid(row = 23, column = 0, sticky="nsw")
    global salt_a
    salt_a = StringVar()
    global salt_a_entry
    salt_a_entry = Entry(root, textvariable=salt_a)
    salt_a_entry.grid(row = 24, column = 0, sticky="nesw")

    global salt_submit
    salt_submit = Button(root, text =' Submit', anchor="w", command = salt_pepper)
    salt_submit.grid(row = 25, column = 0, sticky="nesw")

def salt_pepper():
    global salt_panel
    try:
        salt_panel.grid_remove()
        print("salt_panel removed")
    except:
        print("salt_panel undefined")

    image = plt.imread(filename)
    global img_noise_salt
    img_noise_salt = random_noise(image, mode='s&p',amount=float(salt_a.get()))
    img_noise_salt = np.array(255*img_noise_salt, dtype = 'uint8')

    plt.imshow(img_noise_salt)
    plt.axis('Off')
    image_from_plot = plt_to_img(bbox_inches='tight', pad_inches=0)

    # create a label
    salt_panel = Label(root, image = image_from_plot) 
    # set the image as img
    salt_panel.image = image_from_plot
    salt_panel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

def reset_salt(with_panel=True):
    try:
        salt_a_label.grid_remove()
        print("salt_a_label removed")
    except:
        print("salt_a_label undefined")
    try:
        salt_a_entry.grid_remove()
        print("salt_a_entry removed")
    except:
        print("salt_a_entry undefined")
    try:
        salt_submit.grid_remove()
        print("salt_submit removed")
    except:
        print("salt_submit undefined")
    if with_panel:
        try:
            salt_panel.grid_remove()
            print("salt_panel removed")
        except:
            print("salt_panel undefined")

def periodic():
    global periodic_panel
    try:
        periodic_panel.grid_remove()
        print("periodic_panel removed")
    except:
        print("periodic_panel undefined")

    image = cv2.imread(filename, flags=0)
    shape = image.shape[0], image.shape[1]
    x,y = np.meshgrid(range(0,shape[1]), range(0, shape[0]))
    s= 1+np.sin(x+y/1.5)
    global img_noise_periodic
    img_noise_periodic= ((image) / 128 + s)/4

    plt.imshow(img_noise_periodic, cmap="gray")
    plt.axis('Off')
    image_from_plot = plt_to_img(bbox_inches='tight', pad_inches=0)

    # create a label
    periodic_panel = Label(root, image = image_from_plot) 
    # set the image as img
    periodic_panel.image = image_from_plot
    periodic_panel.grid(row = 14, rowspan = 13, column = 1, padx=10, pady=10)

def reset_periodic():
    try:
        periodic_panel.grid_remove()
        print("periodic_panel removed")
    except:
        print("periodic_panel undefined")

def init_median():
    reset_salt(with_panel=False)

    global median_k_label
    median_k_label = Label(root, text='Kernel Size:')
    median_k_label.grid(row = 23, column = 0, sticky="nsw")
    global median_k
    median_k = StringVar()
    global median_k_entry
    median_k_entry = Entry(root, textvariable=median_k)
    median_k_entry.grid(row = 24, column = 0, sticky="nesw")

    global median_submit
    median_submit = Button(root, text =' Submit', anchor="w", command = median)
    median_submit.grid(row = 25, column = 0, sticky="nesw")

def median():
    global median_panel
    try:
        median_panel.grid_remove()
        print("median_panel removed")
    except:
        print("median_panel undefined")

    global img_noise_salt
    img_cv = cv2.cvtColor(img_noise_salt, cv2.COLOR_RGB2BGR)

    cv2_median = cv2.medianBlur(src=img_cv, ksize=int(median_k.get()))
    cv2_median = cv2.cvtColor(cv2_median, cv2.COLOR_BGR2RGB)
    img_median = resize_img(to_pil(cv2_median))
    img_median = ImageTk.PhotoImage(img_median)

    # create a label
    median_panel = Label(root, image = img_median) 
    # set the image as img
    median_panel.image = img_median
    median_panel.grid(row = 14, rowspan = 13, column = 2, padx=10, pady=10)

def reset_median():
    try:
        median_k_label.grid_remove()
        print("median_k_label removed")
    except:
        print("median_k_label undefined")
    try:
        median_k_entry.grid_remove()
        print("median_k_entry removed")
    except:
        print("median_k_entry undefined")
    try:
        median_submit.grid_remove()
        print("median_submit removed")
    except:
        print("median_submit undefined")
    try:
        median_panel.grid_remove()
        print("median_panel removed")
    except:
        print("median_panel undefined")

def reset():
    reset_original_histogram()    
    reset_equalized_histogram()    
    reset_sobel()
    reset_laplace()
    reset_fourier()
    reset_salt()
    reset_periodic()
    reset_median()

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

btn7 = Button(root, text =' Add Salt And Pepper Noise', anchor="w", command = init_salt_pepper)
btn7.grid(row = 11, column = 0, sticky="nesw")

btn8 = Button(root, text =' Add Periodic Noise', anchor="w", command = periodic)
btn8.grid(row = 12, column = 0, sticky="nesw")

Frame(root, width=300, height=sepFrameHeight).grid(column=0, row = 13)

btn9 = Button(root, text =' Remove Salt And Pepper Noise By Median', anchor="w", command = init_median)
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



