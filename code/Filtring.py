import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageOps, ImageTk, ImageFilter
from tkinter import filedialog, colorchooser
import cv2
import numpy as np
from scipy import stats
from scipy import ndimage
import matplotlib.pyplot as plt
import math


def AddRedColor(file_path, value):
    image = cv2.imread(file_path)
    image = image.astype(np.uint64)
    image[:,:,0] += int(value)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    return image


def SwapGreenRedColor(file_path):
    image = cv2.imread(file_path)
    image[:, :, [1, 0]] = image[:, :, [0, 1]]
    image = Image.fromarray(image)
    return image

def  EliminateRed(file_path):
    image = cv2.imread(file_path)
    image[:,:,0] = 0
    image = Image.fromarray(image)
    return image


def ComplementPhoto(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = 255 - image
    image = Image.fromarray(image)
    return image

def AddPhoto(file_path, value):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.uint64)
    image = image + int(value)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    return image

def SubtractionPhoto(file_path, value):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.int64)
    image = image - int(value)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    return image


def DivisionPhoto(file_path, value):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.int64)
    image = image / int(value)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    return image


def Gray(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    return image

def Stretching(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = 255 * ( (image-image.min()) / (image.max()-image.min()) )
    image = Image.fromarray(image)
    return image


def Equalization(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = Image.fromarray(image)
    return image

def AverageFilter(file_path):
    image = cv2.imread(file_path) 
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])
    averaged = cv2.filter2D(image, -1, kernel) 
    averaged = Image.fromarray(averaged)
    return averaged


def LaplacianFilter(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    kernel=np.array([[1, -2, 1], 
                                [-2,4, -2], 
                                [1, -2,1]]) 

    lablacian=cv2.filter2D(image,-1,kernel)
    # image = image.astype(np.uint8)
    image = Image.fromarray(lablacian)
    return image


def MaximumFilter(file_path):
    image= cv2.imread(file_path)
    imagee =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rd, gn, bl = cv2.split(imagee)
    r = ndimage.maximum_filter(rd, size=9)
    g = ndimage.maximum_filter(gn, size=9)
    b=  ndimage.maximum_filter(bl, size=9)
    result = cv2.merge((r, g, b))
    image=Image.fromarray(result)
    return image
    



def MinimumFilter(file_path):
    image= cv2.imread(file_path)
    imagee =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rd, gn, bl = cv2.split(imagee)
    r = ndimage.minimum_filter(rd, size=9)
    g = ndimage.minimum_filter(gn, size=9)
    b=  ndimage.minimum_filter(bl, size=9)
    result = cv2.merge((r, g, b))
    image=Image.fromarray(result)
    return image


def MedianFilter(file_path):
    image = cv2.imread(file_path)
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp = cv2.medianBlur(temp, 5)
    image = Image.fromarray(temp)
    return image


def AddSaltPepperNoise(file_path, prob=0.02):
    image = cv2.imread(file_path, 0)  # grayscale
    output = np.zeros(image.shape, np.uint8)
    black = 0
    white = 255
    thres=1-prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                output[i][j] = black
            elif rdn > thres:
                output[i][j] = white
            else:
                output[i][j] = image[i][j]
    return Image.fromarray(output)



def RestoreWithAverage(file_path):
    image = cv2.imread(file_path, 0)
    noisy = np.copy(image)
    prob = 0.02
    thres = 1 - prob
    rdn = np.random.random(image.shape)
    noisy[rdn < prob/2] = 0     # Pepper
    noisy[rdn > thres] = 255    # Salt
    
    # Apply average filter
     # تطبيق average filter يدويًا
    h, w = noisy.shape
    restored = np.copy(noisy)

    for i in range(1, h-1):
        for j in range(1, w-1):
            window = noisy[i-1:i+2, j-1:j+2]  # نافذة 3x3
            average_value = np.mean(window)
            restored[i, j] = average_value
    return Image.fromarray(restored.astype(np.uint8))


def RestoreWithMedian(file_path):
    image = cv2.imread(file_path, 0)
    
    # إضافة ضوضاء ملح وفلفل
    noisy = np.copy(image)
    prob = 0.02
    thres = 1 - prob
    rdn = np.random.random(image.shape)
    noisy[rdn < prob/2] = 0     # Pepper
    noisy[rdn > thres] = 255    # Salt

    # تطبيق median filter يدويًا
    h, w = noisy.shape
    restored = np.copy(noisy)

    for i in range(1, h-1):
        for j in range(1, w-1):
            # ناخد نافذة 3x3 حوالين البكسل
            window = noisy[i-1:i+2, j-1:j+2].flatten()
            median_value = np.median(window)
            restored[i, j] = median_value
    return Image.fromarray(restored)
  

def RestoreWithOutlier(file_path,threshold=0.4):
    image = cv2.imread(file_path, 0)
    noisy = np.copy(image)
    prob = 0.02
    thres = 1 - prob
    rdn = np.random.random(image.shape)
    noisy[rdn < prob/2] = 0     # Pepper
    noisy[rdn > thres] = 255 
    output=np.zeros(image.shape, np.uint8)
    kernel = np.array([[1/8, 1/8, 1/8],
                         [1/8,   0, 1/8],
                         [1/8, 1/8, 1/8]], dtype=np.float32)
    
    average=cv2.filter2D(noisy, -1, kernel)
    diff=abs(image-average)
    # Create output image
    output = np.where(diff > threshold, average, image).astype(np.uint8)   
    return Image.fromarray(output)
   
   

def AddGaussianNoise(path):

    img= cv2.imread(path) 
    imagee=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, imagee.shape)
    gauss = gauss.reshape(imagee.shape)
    noisy = imagee + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)
    
def RestoreByImageAveraging(path):
    image = cv2.imread(path)
    imagee=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    noisy_images = []
    for _ in range(5):
        noise = np.random.normal(0, 25, imagee.shape)
        noisy = imagee + noise
        noisy = np.clip(noisy, 0, 255)
        noisy_images.append(noisy)
    avg = np.mean(noisy_images, axis=0).astype(np.uint8)
    return Image.fromarray(avg)

def RestoreByAverageFilter(path):
    image = cv2.imread(path)
    imagee=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, imagee.shape)
    gauss = gauss.reshape(imagee.shape)
    noisy = imagee + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.float32) / 9
    restored = cv2.filter2D(noisy, -1, kernel)
 
    return Image.fromarray(restored)



def BasicGlobalThresholding(path, threshold=127):
    img = cv2.imread(path, 0)
    result = np.zeros_like(img)
    result[img > threshold] = 255
    result[img <= threshold] = 0
    return Image.fromarray(result)


def AutomaticThresholding(path, return_threshold=False):
    img = cv2.imread(path, 0)
    rows, cols = img.shape
    T = np.mean(img)
    T_prev = -1 

    while abs(T - T_prev) >= 0.5:
        T_prev = T
        sum1 = cnt1 = 0        
        sum2 = cnt2 = 0         
        for i in range(rows):
            for j in range(cols):
                pixel = int(img[i, j])
                if pixel >= T:
                    sum1 += pixel
                    cnt1 += 1
                else:
                    sum2 += pixel
                    cnt2 += 1

        if cnt1 == 0 or cnt2 == 0:       
            break

        μ1 = sum1 / cnt1
        μ2 = sum2 / cnt2
        T  = (μ1 + μ2) / 2              
    output = np.where(img >= T, 255, 0).astype('uint8')
    
    if return_threshold:
        return Image.fromarray(output), T
    return Image.fromarray(output)

    
  




def AdaptiveThresholding(path, block_height=50, C=2):
    img = cv2.imread(path, 0)  
    h, w = img.shape
    result = np.zeros_like(img)
    
    for y in range(0, h, block_height):
        y_end = min(y + block_height, h)
        block = img[y:y_end, :]
        mean_value = np.mean(block)
        result[y:y_end, :] = np.where(block > (mean_value - C), 255, 0)

    return Image.fromarray(result)

def SobelEdgeDetection(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise IOError("Image path is incorrect!")
    rows, cols = img.shape
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.int32)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.int32)
    
    G = np.zeros((rows, cols), dtype=np.float32)
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
          
            window = img[y-1:y+2, x-1:x+2].astype(np.int32)
            Gx = np.sum(Kx * window)
            Gy = np.sum(Ky * window)
            G[y, x] = math.sqrt(Gx**2 + Gy**2)  
            # أو |Gx|+|Gy|
    G_max = G.max() if G.max() != 0 else 1
    G_norm = (G / G_max) * 255.0
    sobel_edge = G_norm.astype(np.uint8)
    return Image.fromarray(sobel_edge)



# Helper to read grayscale image
def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

