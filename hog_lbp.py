import numpy as np
import cv2
from skimage.transform import resize
from skimage.feature import hog



readPath = 'captures/'
writePath = 'HOG'

def hog1(gray_img):

    resized_img = resize(gray_img, (64, 64))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    return hog_image

def get_pixel(img, center, x, y):
    new_value = 0

    try:
        if img[x][y] >= center:
            new_value = 1

    except:
        pass

    return new_value


def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = []

    val_ar.append(get_pixel(img, center, x - 1, y - 1))

    val_ar.append(get_pixel(img, center, x - 1, y))

    val_ar.append(get_pixel(img, center, x - 1, y + 1))

    val_ar.append(get_pixel(img, center, x, y + 1))

    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    val_ar.append(get_pixel(img, center, x + 1, y))

    val_ar.append(get_pixel(img, center, x + 1, y - 1))

    val_ar.append(get_pixel(img, center, x, y - 1))

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def lbp1(gray_img):

    resized_img = resize(gray_img, (64, 64))
    height, width= resized_img.shape

    img_lbp = np.zeros((height, width), np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(resized_img, i, j)

    return img_lbp



