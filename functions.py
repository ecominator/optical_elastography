##Important functions##

import numpy as np
import cv2 as cv
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import os
import shutil
from skimage.morphology import skeletonize
import skimage.io

'''Optical Flow'''
'- num_frames: number of frames we want to extract'
'- path_to_source: path to video or sequence of images source'
'- save_folder_name: name of a folder where we want to save the resulting images (only name, the folder itself is to be created by the function)'
'- save (boolean): if we want to save the result or not'
'- color_map (boolean): if we want to create images colored in accordance with the displacement'
'- xy (boolean): if we want to estimate x and y displacements separately'
'- both (boolean): if we do not want to separate x and y displacements'
'- conv_factor: to convert from pixels to micrometers'

def optical_flow (num_frames, path_to_source, conv_factor, save_folder_name=None, save=False, color_map=False, xy=False, both=False):
    cap = cv.VideoCapture(path_to_source) # or ('path to folder with images')
    ret, frame = cap.read()
    prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    if save:
        if xy:
            os.mkdir(save_folder_name+'_x')
            os.mkdir(save_folder_name+'_y')
        if both:
            os.mkdir(save_folder_name)

    result = [] #a list to save the displacements for x and y together
    result_x, result_y = [], [] #lists to save the displacements for x and y separately

    l=1
    while l <= num_frames:
        ret, frame1 = cap.read() #reading a frame
        if not ret: # checking that it is read
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 5, 5, 5, 1.1, 0) #computing the optical flow
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1]) #calculating the magnitude and angle of 2D flow vector
        
        if xy:
            cos = np.cos(ang) #computing sin and cos of the displacement vector angle
            sin = np.sin(ang)
            y_disp = mag*sin
            x_disp = mag*cos
            result_y.append(y_disp * conv_factor)
            result_x.append(x_disp * conv_factor)

            if color_map:
                hsv[..., 0] = ang*180/np.pi/2 #converting to degrees
                hsv_x, hsv_y = np.copy(hsv), np.copy(hsv)
                hsv_y[..., 2] = cv.normalize(mag*sin, None, 0, 255, cv.NORM_MINMAX) #needed to color the displacement map
                hsv_x[..., 2] = cv.normalize(mag*cos, None, 0, 255, cv.NORM_MINMAX)
                bgr_y, bgr_x = cv.cvtColor(hsv_y, cv.COLOR_HSV2BGR), cv.cvtColor(hsv_x, cv.COLOR_HSV2BGR)
                dense_flow_y = cv.addWeighted(frame1, 1, bgr_y, 2, 0) #making a color map
                dense_flow_x = cv.addWeighted(frame1, 1, bgr_x, 2, 0)
            else:
                dense_flow_y = cv.normalize(mag*sin, None, 0, 255, cv.NORM_MINMAX) #computing y displacement
                dense_flow_x = cv.normalize(mag*cos, None, 0, 255, cv.NORM_MINMAX) #computing x displacement
            if save:
                cv.imwrite(save_folder_name+'_y/'+f'{l}.jpg', dense_flow_y)
                cv.imwrite(save_folder_name+'_x/'+f'{l}.jpg', dense_flow_x) 

        if both:
            if color_map:
                hsv[..., 0] = ang*180/np.pi/2 #converting to degrees
                hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) #normalizing the magnitude values
                bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR) #converting back to colored image
                dense_flow = cv.addWeighted(frame1, 1, bgr, 2, 0)
            else:
                dense_flow = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            if save:
                cv.imwrite(save_folder_name+'/'+f'{l}.jpg', dense_flow)
            result.append(dense_flow * conv_factor)
        l = l + 1
        prvs = next
    return result, result_x, result_y

'''Cross-correlation'''

'- num_frames: the number of frames we have extracted'
'- path_to_images: path to the folder with images after OFA'
'- pixel: position of a pixel to correlate'
'- kernel: the size (one integer) of a kernel to perform correlation'
'- x_disp (boolean): if we want to compute cc along x-axis'
'- y_disp (boolean): if we want to compute cc along y-axis'

def cross_correlation (frNum, path_to_images, pixel, kernel, x_disp=False, y_disp=False):
    image_1_name = os.fsdecode(os.listdir(path_to_images)[0])
    image_1 = cv.imread(path_to_images + f'{image_1_name}', cv.IMREAD_GRAYSCALE)
    corr_full_x = np.zeros([image_1.shape[1], frNum])
    corr_full_y = np.zeros([image_1.shape[0], frNum])
    k=0
    while k<frNum:
        image_name = str(k) + '.jpg'
        image_next = cv.imread(path_to_images+f'{image_name}', cv.IMREAD_GRAYSCALE)
        if x_disp:
            arr_1 = np.ones([kernel,]) * image_1[pixel[0], pixel[1]]
            corr = signal.correlate(image_next[pixel[0], :], arr_1, mode='same')
            corr = np.reshape(cv.normalize(corr, None, 1, 256, cv.NORM_MINMAX), (image_1.shape[1],)) #normalization added
            corr_full_x[:, k] = corr
        if y_disp:
            arr_1 = np.ones([kernel,]) * image_1[pixel[0], pixel[1]]
            corr = signal.correlate(image_next[:, pixel[1]], np.transpose(arr_1), mode='same')
            corr = np.reshape(cv.normalize(corr, None, 1, 256, cv.NORM_MINMAX), (image_1.shape[0],)) #normalization added
            corr_full_y[:, k] = corr          
        k+=1
    
    return corr_full_x, corr_full_y


'''Cropping the images''' 

'- data_path: path to the folder with the images to crop'
'- data_save: path to the folder to save the cropped images'
'- x_start: a starting x coordinate to crop an image'
'- x_end: an ending x coordinate to crop an image'
'- y_start: a starting y coordinate to crop an image'
'- y_end: an ending y coordinate to crop an image'
'- color: indicate if the images to crop are colored'

def crop_images(data_path, data_save, x_start, x_end, y_start, y_end, color=False):
    for image in os.listdir(data_path):
        image_name = os.fsdecode(image)
        if color:
            img_next = cv.imread(data_path+f'{image_name}', cv.IMREAD_COLOR)
            cropped_img = img_next[x_start : x_end, y_start : y_end, :]
            cv.imwrite(data_save + f'{image_name}', cropped_img)
        else:
            img_next = cv.imread(data_path+f'{image_name}', cv.IMREAD_GRAYSCALE)
            cropped_img = img_next[x_start : x_end, y_start : y_end]
            cv.imwrite(data_save + f'{image_name}', cropped_img)


'''Simulating a propagating sine wave''' 

'- pixNoX: number of pixels in x direction'
'- pixNoY: number of pixels in y direction'
'- frNum: number of frames'
'- steps: numpy array of step sizes bewteen frames'
'- f: frequency'
'- save_folder_name: name of the folder to save the images'
'- delete_folders: if True, the root folder will be deleted with all its content before running the code'
'- noise: if True, random noise is added'

def sine_wave(pixNoX, pixNoY, frNum, steps, f, save_folder_name, delete_folders=False, noise=False):
    if delete_folders:
        shutil.rmtree(save_folder_name)

    time = np.arange(0, frNum)
    
    x = np.linspace(-100,100,pixNoX)
    y = np.linspace(-50,50,pixNoY)
    xx, yy = np.meshgrid(x,y)
    
    os.mkdir(save_folder_name) # creates root directory
    folders = []
    for step in steps: # iterates over required step sizes
        os.mkdir(save_folder_name + '/' + save_folder_name + f'_{step}') #creates a folder for the waves with a particular step
        folder = os.fsdecode(save_folder_name + '/' + save_folder_name + f'_{step}')
        folders.append(folder)
        images = []
        for i in time:
            if noise:
                noise = np.random.randint(0, 5, size=(pixNoX,pixNoY))
                func = np.sin(2*np.pi*f*(yy[:,0:50])+i*step) + noise[:,0:50]
                images.append(func)
            else:
                func = np.sin(2*np.pi*f*(yy[:,0:50])+i*step)
                images.append(func)
            
        for i in range(frNum):
            img = cv.normalize(images[i], None, 1, 256, cv.NORM_MINMAX) #normalization between 1 and 256
            cv.imwrite(folder + '/' + f'{i}.jpg', img)

    return folders


'''Function to process the images after CC'''

'-threshold_bin: threshold value for binarization (it is better to take something between 130 and 240)'
'-img_after_corr: image obtained after CC'

def processing(threshold_bin, img_after_corr):
    for T in range(threshold_bin, 270, 5): #start with 0
        image = np.copy(img_after_corr)
        image[image<T] = 1 # binarization
        image[image>=T] = 0
        nzero = np.count_nonzero(image) #minimum number of pixels to be considered as a line
        if nzero <= 20:
            continue
        else:
            break
    

    dil_image = skimage.morphology.binary_dilation(image) # dilate 3 times to connect small vertical lines
    dil_image = skimage.morphology.binary_dilation(dil_image)
    dil_image = skimage.morphology.binary_dilation(dil_image)
    skeleton = skeletonize(dil_image) # skeletonize to make the lines thinner
    skeleton = skeleton.astype(np.uint8)

    return skeleton, T           


'''Function to measure a slope'''

'-threshold_bin: threshold value for binarization (it is better to take something between 130 and 240)'
'-img_after_corr: image obtained after CC'

def measure_slope(img_after_corr, threshold_bin):
    img_after_proc, _ = processing(threshold_bin, img_after_corr)
    y = np.nonzero(img_after_proc)[0]
    x = np.nonzero(img_after_proc)[1]

    _, ind, count = np.unique(y, return_index=True, return_counts=True)
    flipped_count = np.flip(count)
    _, ind_count = np.unique(flipped_count, return_index=True)
    if len(ind_count) > 1:
        y = y[ind][0:-ind_count[1]]
        x = x[ind][0:-ind_count[1]]
    else:
        cut, k = 0, 0
        for j in range(len(x)-1):
            if abs((x[j+1]) - x[j]) > 1:
                cut = k
            k+=1
    
        if cut != 0:
            x = np.nonzero(img_after_proc)[1][0:cut]
            y = np.nonzero(img_after_proc)[0][0:cut]

    fit = np.polyfit(x, y, deg = 1)
    slope = np.tan(np.pi - np.arctan(fit[0]))
    ang = np.arctan(slope)*180/np.pi
    
    return slope, ang


'''Implementation of the algorithm for sine waves'''

'- the same parameters as for the functions sine_wave, cross_correlation, and measure_slope'
'- conv_factor: factor to transform from pixels to micrometers'
'- start: should be added if using an old function to measure a slope'

def implementation(pixNoX, pixNoY, frNum, steps, f, save_folder_name, kernel, threshold_bin, conv_factor, delete_folders=False, noise=False):

    #generating sine waves and getting the folder names
    folders = sine_wave(pixNoX, pixNoY, frNum, steps, f, save_folder_name, delete_folders=delete_folders, noise=noise)
 
    #corr_full = np.zeros([pixNoY, frNum, pixNoY, frNum]) #not necessary to keep
    #all_corr_full = []

    slopes = np.zeros([pixNoY, frNum])
    angs = np.zeros([pixNoY, frNum])
    all_slopes = np.zeros([len(steps), pixNoY, frNum])
    all_angs = np.zeros([len(steps), pixNoY, frNum])
    for folder, k in zip(folders, range(0, len(steps)+1)):
        path_to_images = f'{folder}/'
        for i in range(pixNoY):
            for j in range(frNum):
                pixel=[i, j]
                _, corr_full_y = cross_correlation(frNum, path_to_images, pixel, kernel, x_disp=False, y_disp=True)
                #corr_full[i][j] = corr_full_y
                img_after_corr = corr_full_y
                slope, ang = measure_slope(img_after_corr, threshold_bin)
                slopes[i, j] = slope*conv_factor
                angs[i, j] = ang
        all_slopes[k] = slopes
        all_angs[k] = angs
        #all_corr_full.append(corr_full)

    print('The slopes and angles for propagating sine waves are:')
    for i in range(0, len(steps)):
        print(f'Step size 0.{i+1}:', round(all_slopes[i, 0, 0], 2),'/', round(all_angs[i, 0, 0], 2), '\n')

    #printing the result

    all_slopes_1 = all_slopes[0]
    for i in range(1, len(steps)):
        all_slopes_big = np.concatenate((all_slopes_1, all_slopes[i]), axis=1)
        all_slopes_1 = all_slopes_big

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(all_slopes_big)
    plt.title('Slopes for the sine waves propagating with increasing rate')
    plt.colorbar()
    plt.show()

    return all_slopes, all_angs


'''Width calculation from CC images'''

'- frNum, path_to_images, pixel, kernel: for cross-correlation' 
'- threshold_bin: threshold value for binarization'
'- alpha: frNum/alpha is a threshold for the number of nonzero pixels in a row'
'- show(boolean): if plotting of images after cc and binarization is required'
'- conv_factor: to convert from pixels to micrometers'

def width_calculation(frNum, path_to_images, pixel, kernel, threshold_bin, alpha, conv_factor, show=False):
    _, corr_full_y = cross_correlation(frNum, path_to_images, pixel, kernel, x_disp=False, y_disp=True)
    T = threshold_bin
    image = np.copy(corr_full_y)
    image[image<T] = 1 # binarization and inversion
    image[image>=T] = 0

    if show:
        fig, axs = plt.subplots(1, 2, figsize=[6, 4])
        axs[0].imshow(corr_full_y, cmap='binary')
        axs[0].set_title('Image after CC')

        axs[1].imshow(image, cmap='binary')
        axs[1].set_title('Binarized and inversed image after CC')

        plt.show()

    upper_width = 0
    lower_width = 0


    for i in range(pixel[0], corr_full_y.shape[0], 1):
        width = np.count_nonzero(image[i, :])
        if width >= np.floor(frNum/alpha):
            upper_width+=1
        else:
            break

    for i in range(pixel[0]-1, 0, -1):
        width = np.count_nonzero(image[i, :])
        if width >= np.floor(frNum/alpha):
            lower_width+=1
        else:
            break

    return (upper_width + lower_width) * conv_factor 

'''Elasticity calculation from the width calculation function'''

'- width: output of width calculation in um'
'- freq: applied frequency in Hz'
'- dens: medium density in kg/m^3'

def elasticity_width (width, freq, dens):
    width_m = width * (10**(-6))
    mu = dens * (2*width_m)**2 * freq**2
    return mu


'''Slope calculation using Hough Transform'''

'-img_after_corr: image after applying cross-correlation'

def measure_slope_HT (img_after_corr):
    corr = np.uint8(img_after_corr)
    dst = cv.Canny(corr, 50, 200, None, 3)
    # plt.imshow(dst)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 40)

    cdstP = np.zeros_like(dst)
    funs = []
    coefs = []

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv.LINE_AA)

            x = [l[0], l[2]]
            y = [-l[1], -l[3]]
            coef = np.polyfit(x, y, deg = 1)
            fun = np.poly1d(coef)
            coefs.append(coef)
            funs.append(fun)
    
    # plt.imshow(cdstP)
    # x=np.linspace(0, 10, 10)
    # fig = plt.figure(figsize=(1, 10.4))
    # for fun in funs:
    #     plt.plot(x, fun(x))
    # plt.show()

        slope = np.mean(coefs, axis=0)[0]
        ang = 180 + np.arctan(slope)*180/np.pi #to change for sines

    else:
        coefs, slope, ang = None, None, None

    return coefs, slope, ang

'''Elasticity calculation'''

'- slopes: computed slopes'
'- dens: medium density'
'- conv_pix: factor to convert from pixels to micrometers'
'- conv_fr: factor to convert from frames/sec to 1/sec'

def elasticity (slopes, dens, conv_pix, conv_fr):
    conv_factor = (conv_pix * conv_fr) * 10**(-6)
    mu = (slopes * conv_factor)**2 * dens
    return mu


'''Skipping frames'''

'- path_to_frames: path to a folder with the frames'
'- step: number of frames to skip between two consecutive frames + 1'

def skip_frames (path_to_frames, step):
    new_path = path_to_frames + f'_slow_{step}'
    isdir = os.path.isdir(new_path)

    if isdir is False:
        os.mkdir(new_path)
    
    for i in range(0, len(os.listdir(path_to_frames)), step):
        img = cv.imread(path_to_frames+f'/{i}.jpg')
        cv.imwrite(new_path+f'/{int(i/step)}.jpg', img)
    return new_path