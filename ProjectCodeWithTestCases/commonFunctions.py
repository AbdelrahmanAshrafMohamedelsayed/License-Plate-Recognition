import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram 
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv 

# Convolution:
from scipy.signal import convolve2d 
from scipy import fftpack
import math
import joblib 

import cv2 
import cv2 as cv
from skimage.util import random_noise 
from skimage.filters import median 
from skimage.feature import canny 
from skimage.measure import label 
from skimage.color import label2rgb

from mpl_toolkits.mplot3d import Axes3D  

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin # 
from skimage.measure import find_contours 
from skimage.draw import rectangle

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    

def show_3d_image(img, title):
    fig = plt.figure()
    fig.set_size_inches((12,8))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, img.shape[0], 1)
    Y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = img[X,Y]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(title)
    plt.show()
    
def show_3d_image_filtering_in_freq(img, f):
    img_in_freq = fftpack.fft2(img)
    filter_in_freq = fftpack.fft2(f, img.shape)
    filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)
    
    img_in_freq = fftpack.fftshift(np.log(np.abs(img_in_freq)+1))
    filtered_img_in_freq = fftpack.fftshift(np.log(np.abs(filtered_img_in_freq)+1))
    
    show_3d_image(img_in_freq, 'Original Image')
    show_3d_image(filtered_img_in_freq, 'Filtered Image')


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')
    
    

class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        #assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape[0], self.img.shape[1]
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

def read_img(img):
    # im= cv2.imread(img_path)
    im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray,(15,15), 0)
    roi= cv2.resize(im_gray,(28,28), interpolation=cv2.INTER_AREA)
    
    rows, cols = roi.shape
    
    ## Add pixel one by one into data array
    for i in range(rows):
        for j in range(cols):
            k =roi[i,j]
            if k>100:
                k=1
            else:
                k=0
    
        hog = Hog_descriptor(roi, cell_size=4, bin_size=9)
        vector, image = hog.extract()
        vec = np.array(vector)
        return vec.flatten()


model=joblib.load("./model/char_rec")


def Sort_items(sub_li):
        return(sorted(sub_li, key = lambda x: x[1]))   
def proj(path_name):
        img = io.imread(path_name)[...,:3]
        io.imshow(img)
        gray =(rgb2gray(img)*255).astype(np.uint8)
        (T, threshInv) = cv.threshold(gray, 115, 255,cv.THRESH_BINARY_INV)
        kernel = np.ones((2,2),np.uint8)
        erosion = cv.dilate(threshInv,kernel,iterations = 1)
        can2 = canny(threshInv, sigma=0.5, low_threshold=100, high_threshold=200)
        cnts2,new2 = cv.findContours(can2.copy().astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        image12=img.copy()
        cv.drawContours(image12,cnts2,-1,(0,255,0),3)
        # cnts = sorted(cnts, key = cv.contourArea, reverse = True) [:20]
        # image1=img.copy()
        # cv.drawContours(image1,cnts,-1,(0,255,0),3)
        cnts2 = sorted(cnts2, key = cv.contourArea, reverse = True) 
        image12=img.copy()
        cv.drawContours(image12,cnts2,-1,(0,255,0),3)
        plate = None
        new_img=None
        hieght_of_licence=0
        width_of_licence= 0
        i=0
        for c in cnts2:
                x,y,w,h = cv.boundingRect(c) 
                plate=img[y:y+h,x:x+w]
                i+=1
                perimeter = cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, 0.018 * perimeter, True)
                height, width = image12.shape[:2]
                gray55 =(rgb2gray(plate)*255).astype(np.uint8)
                can25 = canny(gray55, sigma=1, low_threshold=120, high_threshold=200)
                cnts4,new2 = cv.findContours(can25.copy().astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                if(89978>=w*h >14000 and w>h and (2 <= w/h <= 4) ): 
                        perimeter = cv.arcLength(c, True)
                        approx = cv.approxPolyDP(c, 0.018 * perimeter, True)
                        if len(approx) == 4: 
                                screenCnt = approx
                                x,y,w,h = cv.boundingRect(c) 
                                new_img=img[y:y+h,x:x+w]
                                show_images([new_img],[path_name])
                                hieght_of_licence=h
                                width_of_licence=w
                                # print(' accurcy 80%')
                                break
                        else:
                                screenCnt = approx
                                x,y,w,h = cv.boundingRect(c) 
                                new_img=img[y:y+h,x:x+w]
                                show_images([new_img],[path_name])
                                hieght_of_licence=h
                                width_of_licence=w
                                # print('accurcy 60%')
                                break
        if hieght_of_licence>0:
                # print(hieght_of_licence)
                gray_letters =(rgb2gray(new_img)*255).astype(np.uint8)
                (T, threshInv) = cv.threshold(gray_letters, 115, 255,cv.THRESH_BINARY_INV)
                can8 = canny(threshInv, sigma=0.5, low_threshold=100, high_threshold=200)
                cnts4,new2 = cv.findContours(can8.copy().astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                image12=new_img.copy()
                cv.drawContours(image12,cnts4,-1,(0,255,0),3)
                arr_of_letter = []
                for c2 in cnts4:
                        perimeter = cv.arcLength(c2, True)
                        approx = cv.approxPolyDP(c2, 0.018 * perimeter, True)
                        x,y,w,h = cv.boundingRect(c2) 
                        new_img2=new_img[y:y+h,x:x+w]
                        if(hieght_of_licence * 0.53 >h> hieght_of_licence*0.22 and width_of_licence*.01<w<width_of_licence*.40 ): 
                                arr_of_letter.append([new_img2,x])
                sorted_letters=Sort_items(arr_of_letter)
                plate_chars = ""
                for img in sorted_letters:
                        # show_images([img[0]],['l'])
                        hogTest = read_img(img[0])
                        predictions  =model.predict([hogTest])
                        plate_chars+=predictions[0]
                print(plate_chars)
                return plate_chars

def getAccuracy(testSet, predictions):
        correct = 0
        total = 0
        for x in range(len(testSet)):
            total += len(testSet[x])
            for i in range(len(predictions[x])):
                if testSet[x].find(predictions[x][i]) != -1:
                        correct += 1
        return (correct/float(total)) * 100.0
    
def getFullAccuracy(testSet, predictions):
        correct = 0
        total = len(testSet)
        for x in range(len(testSet)):
            if predictions[x].find(testSet[x]) != -1:
                correct += 1
        return (correct/float(total)) * 100.0

