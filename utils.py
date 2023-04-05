import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def fill (src_img, offset = 2):
    img_filled = src_img.copy()
    height, width = img_filled.shape
    mask = np.zeros((height+offset, width+offset), np.uint8)
    cv.floodFill(img_filled, mask, (0,0), 255)
    #invert the image
    img_filled_inv = cv.bitwise_not(img_filled)
    #combine the two images to extract the forground
    return src_img | img_filled_inv

def crop(img_treat, img_original, offset = 10):
    coordinates = cv.findNonZero(img_treat)
    x, y, w, h = cv.boundingRect(coordinates)
    h_image, w_image = img_treat.shape
    y_lower = y - offset
    y_upper = y + h + offset
    x_lower = x - offset
    x_upper = x + w + offset
    
    if(y_lower<0):
        y_lower = 0
    if(y_upper>h_image):
        y_upper = h_image
    if(x_lower<0):
        x_lower=0
    if(x_upper>w_image):
        x_upper = w_image
    return img_treat[y_lower:y_upper, x_lower:x_upper], img_original[y_lower:y_upper, x_lower:x_upper],\
        x_lower, y_lower

def resize_percent(image, percent):
    width = int(image.shape[0] * percent/100)
    height = int(image.shape[1] * percent/100)
    return cv.resize(src = image, dsize=(width, height), interpolation = cv.INTER_LINEAR)

def centroid_calc(image):
    M = cv.moments(image)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def edge_detect(image, th1 = 127, th2 = 255, aperture_size =3 , l2_gradient = True):
    return cv.Canny(image = image, threshold1 = th1, threshold2 = th2, apertureSize = aperture_size, L2gradient = l2_gradient)

def circle_detect(img, dp=5, param1=255, param2=220, offset = 5):
    image_temp = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    height, width = image_temp.shape[:2]
    circles = cv.HoughCircles(image = img, method = cv.HOUGH_GRADIENT, dp = dp, minDist = int(width/4), param1=param1,
         param2=param2, minRadius = int(0.9*width/2), maxRadius = int(width/2))
    
    #convert the x,y and radius of the circles to integer
    if circles is not None:
        circles = np.round(circles[0,:]).astype("int")
        for (x,y,r) in circles:
            cv.circle(img = image_temp, center = (x,y), radius = r, color = (255, 0, 0), thickness = 2)
            cv.rectangle(img = image_temp, pt1 = (x-offset, y-offset), pt2 = (x+offset, y+offset),
                color = (255, 0, 0), thickness = -1)
            center_x = int(np.mean(circles[:,0]))
            center_y = int(np.mean(circles[:,1]))
            return image_temp, center_x, center_y
    else:
        return None, None, None

def line_detect(img, rho=1, theta=1, threshold=60, minLineLength=120, maxLineGap=50, min_slope=65, max_slope=75):
    image_temp = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    lines = cv.HoughLinesP(image_temp, rho=rho, theta=theta*np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if min_slope <= np.arctan2(abs(y2-y1)/abs(x2-x1))/np.pi <= max_slope:
                points = [x1, y1, x2, y2]
                cv.line(img=image_temp, pt1 = (x1,y1), pt2 = (x2,y2), color=(255,0,0), thickness=2)
                return image_temp, points
            return None, None

def line_detection(image_temp, rho=1, theta=1, threshold=60, minLineLength=120, maxLineGap=50, min_slope=65, max_slope=75):
    #image_temp = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    lines = cv.HoughLinesP(image_temp, rho=rho, theta=theta*np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if min_slope <= np.arctan2(abs(y2-y1),abs(x2-x1))/np.pi <= max_slope:
                points = [x1, y1, x2, y2]
                #image_temp = cv.cvtColor(image_temp, cv.COLOR_GRAY2RGB)
                cv.line(img=image_temp, pt1 = (x1,y1), pt2 = (x2,y2), color=(255,0,0), thickness=2)
                return image_temp, points
            return None, None

def count_gray_levels(img):
    unique, counts = np.unique(img, return_counts=True)
#     plt.plot(unique, counts)
#     plt.show()
#     return int(unique[np.argmax(counts)])
    return int(np.max(unique))


def show_image(img, cmap='gray'):
    if cmap == 'rgb':
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()




