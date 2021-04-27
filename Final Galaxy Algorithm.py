# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:35:37 2021

@author: Nathalie
"""


from astropy.io import fits
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.utils.data import get_pkg_data_filename
import math
from sklearn.linear_model import LinearRegression


hdulist = fits.open("A1_mosaic.fits")

header = hdulist[0].header
data = hdulist[0].data

image_file = get_pkg_data_filename('A1_Mosaic.fits')
image_data = fits.getdata(image_file, ext=0)
plt.clf()
plt.imshow(image_data)
plt.show()

image = image_data[100: 400, 1050: 1750]

mean_background = 3.41869525e+03 #Mean background for the whole image
background_stdev = 1.21961107e+01 #Background standard deviation for the whole
                                    #image
min_radius_4s = 3.77828618968386 #Minimum characteristic radius at 4 sigma (s)
min_radius_3pt5s = 1.9946483113443867 #Minimum characteristic radius at 3.5s
min_radius_3pt5s_stdev = 2.0517556677065034 #Standard dev of min radius at 3.5s

#Galaxy Radii for 3.5 standard deviations above mean background.
Minimum_Radius =  2.022293187264448
Minimum_Radius_Standard_Deviation =  2.0650153030816663

def Locus(image_data):
    """Finds the index of the brightest galaxy once filters are applied to 
    input data at its stage of filtration
    
    INPUTS
    ------
    data = 2D array containing the number of counts in each pixel
    
    OUTPUTS
    -------
    [0]: data[index] = number of counts in the brightest point in the image
    [1]: index = index of the brightest point in the image"""
    
    index = sp.unravel_index(sp.argmax(image_data), image_data.shape)
    return [data[index], index]

def Gaussian(x, amplitude, mean, stddev):
    """Equation of a Gaussian function
    
    INPUTS
    ------
        x = independent variable
        amplitude = height of the Gaussian
        mean = x-value at the peak of the Gaussian
        stddec = standard deviation
        
    OUTPUTS
    -------
        A Gaussian probability density function of x"""
    return amplitude*np.exp(-(x-mean)**2/(2*(stddev**2)))

def histogram1D(image_data = image_data, lower_fit = 3300, upper_fit = 3800):
    """Prints a histogram of the pixel counts and fits a Guassian to the 
    plot.
    
    INPUTS
    ------
    image_data = 2D array containing the number of counts in each pixel
    lower_fit = lower bound of the plotted histogram and Gaussian curve fit
    upper_fit = upper bound of the plotted histogram and Gaussian curve fit"""
    
    
    
    data = np.concatenate(image_data)
 
    x = np.arange(lower_fit, upper_fit+1)
    y = np.zeros(len(x))
    i = 0
    while i < len(data):
        if data[i] <= upper_fit and data[i] >= lower_fit:
            y[data[i] - lower_fit] += 1
        i += 1
    print(y)
    
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    
    popt,pcov = curve_fit(Gaussian, x, y, p0=[max(y), mean, sigma])

    print (x)
    print (max(y))
    print(len(data))
    print ('Gaussian Fit Parameters:', popt, pcov)
    
    mean_background = popt[1]
    background_stdev = popt[2]
    
    plt.figure()
    plt.hist(data, range = (lower_fit, upper_fit), 
             bins = int(upper_fit-lower_fit), label = 'Pixel Counts')
    plt.plot(x, Gaussian(x, *popt), 'r-', label='Gaussian Fit')
    plt.legend()
    plt.xlabel('Number of Counts')
    plt.ylabel('Number of Pixels')
    plt.title("Distribution of counts in pixels")
    
    return mean_background, background_stdev 

    

def Initial_Filter(image_data):
    """Perform initial filter of the image. Mask artifacts around the edges of
    the image, as well as the bright star in the centre with bleeding pixels.
    
    INPUTS
    ------
        image_data = 2D array containing the number of counts in each pixel
        
    OUTPUT
    ------
        [0]: new_image = an updated 2D array containing the number of counts in
            each pixel, with masking applied to artifacts.
        [1]: Filter_Shape = an array of 1s and 0s containing all of the points 
            in the image that have been masked."""
    
    #Copy of the original image
    new_image = np.ones_like(image_data) * image_data
    
    #Cutting out the top border
    top_backg = np.median(image_data[100:200, 100:image_data.shape[1]-100])
    
    for i in np.arange(0 , 101):
        for j in np.arange(0, image_data.shape[1]):
            new_image[i][j] = top_backg
            
    #Cutting out the left border
    left_backg = np.median(image_data[100:image_data.shape[0], 100:200])
    #print ("left_backg = ", left_backg)
    for i in np.arange(0,image_data.shape[0]):
        for j in np.arange(0, 101):
            new_image[i][j] = left_backg
            
    #Cutting out the right border
    right_backg = np.median(image_data[100:image_data.shape[0]-100, 
                                       image_data.shape[1]-200:\
                                       image_data.shape[1]-100])

    for i in np.arange(0, image_data.shape[0]):
        for j in np.arange(image_data.shape[1]-100, image_data.shape[1]):
            new_image[i][j] = right_backg
            
    #Cutting out the bottom border
    bottom_backg = np.median(image_data[image_data.shape[0]-200:\
                                        image_data.shape[0]-100, 100:\
                                        image_data.shape[1]-100])
    
    for i in np.arange(image_data.shape[0]-100, image_data.shape[0]):
        for j in np.arange(0, image_data.shape[1]):
            new_image[i][j] = bottom_backg
    
    #Cutting out the bleeding pixels near the top of the image
    artifacts_backg = (np.median(image_data[100: 300, 1050: 1300]) + 
                 np.median(image_data[100: 300, 1650: 1750]))/2
    
    for i in np.arange(100, 501):
        for j in np.arange(1050, 1751):
            new_image[i][j] = artifacts_backg
            
    #Cutting out the giant bleeding star in the centre
    y = 3211
    x = 1436
    
    c = 320
    
    ##plt.imshow(new_image)
    ##plt.show()
    
    #print ("Brightest spot = ", new_image[y][x])
    
    
    star_backg = (np.median(image_data[0:image.shape[0], (x-2*c):(x-c)]) + 
                    np.median(image_data[0:image.shape[0], (x+c):(x+2*c)]))/2
    
    #print ("Background = ", star_backg)
                  
    #Masking the main body of the central star with a circle
    i,j = np.ogrid[-y:new_image.shape[0]-y, -x:new_image.shape[1]-x]
    for i in np.arange(-c, c):
        for j in np.arange(-c, c):
            if (i**2) + (j**2) <= c**2:   
                new_image[i+y][j+x] = star_backg
    
    
    #Masking the rest of the bleeding pixels in the star with an ellipse
    x = 1436
    y = 3211
    
    a = 3211
    b = 15
    
    
    i,j = np.ogrid[-y:new_image.shape[0]-y, -x:new_image.shape[1]-x]
    
    mask = ((i**2/a**2) + (j**2/b**2) <= 1) + ((i**2) + (j**2) <= c**2) + \
    (i+y <= 100)+ (i+y >= new_image.shape[0]-100) + (j+x <= 100) + \
    (j+x >= new_image.shape[1] - 100) + ((i+y <= 500) & (j+x <=1750) & \
     (j+x >= 1050))
    
    filter_shape = np.ones_like(new_image)
    filter_shape = filter_shape * mask
    
    plt.imshow(filter_shape)
    plt.show()
    for i in np.arange(-y, new_image.shape[0]-y):
        for j in np.arange(-b, b):
            if (i/a)**2 + (j/b)**2 <= 1:   
                new_image[i+y][j+x] = star_backg
    
    #print ("Star Centre = ", (y,x))
    
    ##plt.imshow(new_image)
    ##plt.show() 

    
           
    return new_image, filter_shape


def StarMask (index, image, Filter, thresh = 2, c = 50, ellipticity_limit = 1):
    """Identifies whether a bright spot is part of a star with bleeding pixels 
    in the image. Returns a new image that masks the star without counting it.
    
    INPUTS
    ------
        index = index of the brightest point in the image
        image = 2D array containing the counts in each pixel
        Filter = 2D array of 1s and 0s containing all of the points in the
            image that have been masked up to this point
        thresh = number of standard deviations away from the median beackground
            at which the mask is applied (Default is 2)
        c = Number of pixels away from the brightest point at which the 
            background is calculated (Default is 50)
        ellipticity_limit = the ratio of the major to minor axes down to which 
            a star with bleeding pixels is identified
        
            
    OUTPUTS
    -------
        [0]: IsStar = Boolean variable determining whether or not the bright 
            spot contains bleeding pixels
        [1]: new_image = 2D array in which the bright spot is masked with the 
            median local background
        [2]: Filter = 2D array of 1s and 0s containing all of the points in the
            image that have been masked up to this point"""
    
    y = index[0]
    x = index[1]
    
    #plt.clf()
    #plt.imshow(image[y-100:y+100, x-100:x+100])
    #plt.show()
    
    #Make a copy of the filter matrix
    New_Filter = np.ones_like(Filter)*Filter
    
    #Define the median background and upper threshold
    star_backg = (np.median(image[y-c:y+c, (x-2*c):(x-c)]) + 
                    np.median(image[y-c:y+c, (x+c):(x+2*c)]))/2
    
    threshold = star_backg + thresh*(np.std(image[(y-c):(y+c), (x-2*c):(x-c)]) + 
                    np.std(image[(y-c):(y+c), (x+c):(x+2*c)]))/2
       
    #print ("Background threshold = ", threshold)             
    #print ("Star Background = ", star_backg)               
    
    #Make a copy of the image matrix
    new_image = image * np.ones_like(image)
    
    #Measure the width of the star
    if x >= new_image.shape[1]/2:
        for r in np.arange(1, new_image.shape[1]-x):
            median = np.median(image_data[y][x-r:x+r])
            if median < threshold:
                b = r
                #print ("Median = ",np.median(image_data[y][x-r:x+r]))
                #print ("b = ", b)
                break
            b = r
    else:
        for r in np.arange(1, x):
            median = np.median(image_data[y][x-r:x+r])
            if median < threshold:
                b = r
                #print ("Median = ",np.median(image_data[y][x-r:x+r]))
                #print ("b = ", b)
                break
            b = r

    #Measure the height of the star   
    if y >= new_image.shape[0]/2:
        for r in np.arange(1, new_image.shape[0]-y):
            image_transpose = new_image.transpose()
            median = np.median(image_transpose[x][y-r:y+r])
            if median < threshold:
                a = r
                #print ("Median = ",np.median(image_transpose[x][y-r:y+r]))
                #print ("a = ", a)
                break
            a = r
    else:
        for r in np.arange(1, y):
            image_transpose = new_image.transpose()
            median = np.median(image_transpose[x][y-r:y+r])
            if median < threshold:
                a = r
                #print ("Median = ",np.median(image_transpose[x][y-r:y+r]))
                #print ("a = ", a)
                break
            a = r
    
    
    #print ("Star axes = ", (a,b))
    
    #Identify whether there are bleeding pixels spanning vertically above and 
    #below the bright spot
    if a <= ellipticity_limit*b:
        IsStar = False
    else:
        IsStar = True
    
    #Top left section of ellipse is cut off
    if y-a < 0 and x-b < 0:
        for i in np.arange(-y, a):
            for j in np.arange(-x, b):
                if (i/a)**2 + (j/b)**2 <= 1:
                    new_image[i+y][j+x] = star_backg
                    New_Filter[i+y][j+x] = 1
    
    #Top right section of ellipse is cut off
    elif y-a < 0 and x+b >= new_image.shape[1]:
        for i in np.arange(-y, a):
            for j in np.arange(-x, b):
                if (i/a)**2 + (j/b)**2 <= 1:
                    new_image[i+y][j+x] = star_backg
                    New_Filter[i+y][j+x] = 1
    
    #Bottom left section of ellipse is cut off:
    elif y+a >= new_image.shape[0] and x-b < 0:
        for i in np.arange(-a, new_image.shape[0]-y):
            for j in np.arange(-x, b):
                if (i/a)**2 + (j/b)**2 <= 1:
                    new_image[i+y][j+x] = star_backg
                    New_Filter[i+y][j+x] = 1
                    
    #Bottom right section of ellipse is cut off:
    elif y+a >= new_image.shape[0] and x+b >= new_image.shape[1]:
        for i in np.arange(-a, new_image.shape[0]-y):
            for j in np.arange(-b, new_image.shape[1]-x):
                if (i/a)**2 + (j/b)**2 <= 1:
                    new_image[i+y][j+x] = star_backg
                    New_Filter[i+y][j+x] = 1
                
    #Left section of ellipse is cut off:
    elif x-b < 0:
        for i in np.arange(-a, a):
            for j in np.arange(-x, b):
                if (i/a)**2 + (j/b)**2 <= 1:
                    new_image[i+y][j+x] = star_backg
                    New_Filter[i+y][j+x] = 1
                
    #Top section of ellipse is cut off:
    elif y-a < 0:
        for i in np.arange(-y, a):
            for j in np.arange(-b, b):
                if (i/a)**2 + (j/b)**2 <= 1:
                    new_image[i+y][j+x] = star_backg 
                    New_Filter[i+y][j+x] = 1
                    
    #Bottom section of ellipse is cut off:
    elif y+a >= new_image.shape[0]:
        for i in np.arange(-a, new_image.shape[0]-y):
            for j in np.arange(-b, b):
                if (i/a)**2 + (j/b)**2 <= 1:
                    new_image[i+y][j+x] = star_backg
                    New_Filter[i+y][j+x] = 1
    
    #Right section of ellipse is cut off:
    elif x+b >= new_image.shape[1]:
        for i in np.arange(-a, a):
            for j in np.arange(-b, new_image.shape[1]-x):
                if (i/a)**2 + (j/b)**2 <= 1:
                    new_image[i+y][j+x] = star_backg
                    Filter[i+y][j+x] = 1
    
    else:
        for i in np.arange(-a, a):
            for j in np.arange(-b, b):
                if (i/a)**2 + (j/b)**2 <= 1:   
                    new_image[i+y][j+x] = star_backg
                    New_Filter[i+y][j+x] = 1
    
    #plt.clf()
    #plt.imshow(new_image[y-100:y+100, x-100:x+100])
    #plt.show()
    
    return IsStar, new_image, New_Filter
    
    
def Galaxy(index, image, Filter, lower = 25, upper = 40, thresh =3, 
           max_radius = 110):
    """Using the brightest spot in the image, this function calculates the 
    median value of the local background, and its standard deviation. It then 
    calculates the mean value of a ring of pixels around the brightest point 
    until it reaches a threshold value above the median background, thereby 
    determining the radius of the galaxy. It then uses this radius to create an
    aperture around the brightest point which is used to sum up the total 
    number of counts within the galaxy. Finally, a mask equivalent to the 
    median of the local background is applied to the image in the shape of the
    aperture, and this same shape of 1s is added to the input filter matrix. If
    part of the aperture is cut off by a filter applied previously to the image
    or an edge of the image, then the number of counts in the galaxy is set to 
    0 and the image is masked and filter appended as normal.
    
    INPUTS
    ------
        index = index of the brightest pixel in the image
        image = 2D array containing the number of counts in each image
        Filter = 2D array of 1s and 0s containing the shape of the mask that 
            has been previously applied to the image
        lower = Lower radius of the annulus used to calculate the local 
            background (Set to 25)
        upper = Upper radius of the annulus used to calculated the local
            background (Set to 40)
        thresh = number of standard deviations above the median local
            background up to which the aperture and mask is applied
            
    OUTPUTS
    -------
        
        [0]: r = radius of the galaxy
        [1]: local_backg = median value of the local background
        [2]: backg_err = standard deviation of the local background
        [3]: galaxy_counts = number of counts in the galaxy
        [4]: counts_err = error on the number of counts in the galaxy
        [5]: new_image = updated 2D array containing the number of counts in
            each pixel
        [6]: Filter = updated 2D of 1s and 0s containing the shape of the mask
            that has been applied to the image up to this point"""
    
    
    #Make a copy of the filter matrix
    New_Filter = np.ones_like(Filter)*Filter
    
    #Define Image boundaries
    xmax = image.shape[0]
    ymax = image.shape[1]
   
    #Define centres
    a = index[0]
    b = index[1]
    y,x = np.ogrid[-a:xmax-a, -b:ymax-b]
    
    plt.clf()
    ##plt.imshow(image[a-100:a+100, b-100:b+100])
    plt.show()
    
    #Define annulus used to calculate background
    annulus = (x*x + y*y >= lower*lower) & (x*x + y*y <= upper*upper)
    #plt.imshow(annulus[a-100:a+100, b-100:b+100])
    #plt.show()
    #print("Local background = ", np.median(image[annulus]))
    local_backg = np.median(image[annulus])
    backg_err = np.std(image[annulus])
   
    #Define threshold to be "thresh" number of standard deviations away from 
    #the mean background
    threshold = local_backg + thresh*backg_err
    #print ("Brightest point = ", image[index])
    #print ("Threshold = ", threshold)
    
    galaxy_counts = 0
    
    
    #Calculating mean values of rings growing outwards from the locus, until
    #background threshold is reached.
    if math.isnan(local_backg) == False:
        for r in range(1, max_radius):
            testring = (x*x + y*y == r*r)
            #plt.imshow(testring[a-100:a+100, b-100:b+100])
            #plt.show()
            #print ("Testring mean = ", np.mean(image[testring]))
            if np.mean(image[testring]) < threshold:
                aperture = np.zeros_like(image)
                new_image = np.ones_like(image) * image
                pixels = 0
                
                
                #Top left section of galaxy is cut off
                if a-r < 0 and b-r < 0:
                    #print("Top left corner")
                    for x in np.arange(-b, r):
                        for y in np.arange(-a, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter

                #Bottom left corner of galaxy is cut off
                elif b-r < 0 and a+r >= image.shape[0]:
                    #print ("Bottom left corner")
                    for x in np.arange(-b, r):
                        for y in np.arange(-r, image.shape[0]-a):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Top and right of galaxy are cut off
                elif b+r >= image.shape[1] and a-r <0:
                    #print("Top right corner")
                    for x in np.arange(-r, image.shape[1]-b):
                        for y in np.arange(-a, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
               
                #Bottom right of galaxy is cut off
                elif a+r >= image.shape[0] and b+r >= image.shape[1]:
                    #print ("Bottom right corner")
                    for x in np.arange(-r, image.shape[1]-b):
                        for y in np.arange(-r, image.shape[0]-a):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Right side of galaxy is cut off
                elif b+r >= image.shape[1]:
                    #print ("Right edge")
                    for x in np.arange(-r, image.shape[1]-b):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter        
                
                #Bottom of galaxy is cut off
                elif a+r >= image.shape[0]:
                    #print("Bottom edge")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, image.shape[0]-a):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image , New_Filter         
                
                #Top of galaxy is cut off
                elif a-r < 0:
                    #print("Top edge")
                    for x in np.arange(-r, r):
                        for y in np.arange(-a, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter

                #Left of galaxy is cut off
                elif b-r < 0:
                    #print("Left edge")
                    for x in np.arange(-b, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter        
                
                #Top left section of galaxy in filter
                elif Filter[a-r][b-r] == 1:
                    #print("Top left corner in filter")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Top right section of galaxy in filter
                elif Filter[a-r][b+r] == 1:
                    #print("Top right corner in filter")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Bottom left section of galaxy in filter
                elif Filter[a+r][b-r] == 1:
                    #print("Bottom left corner in filter")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Bottom right section of galaxy in filter
                elif Filter[a+r][b+r] == 1:
                    #print("Bottom right corner in filter")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Left section of galaxy in filter
                elif Filter[a][b-r] == 1:
                    #print("Left side in filter")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Top section of galaxy in filter
                elif Filter[a-r][b] == 1:
                    #print("Top side in filter")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Bottom section of galaxy in filter
                elif Filter[a+r][b] == 1:
                    #print("Bottom side in filter")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
                #Right section of galaxy in filter
                elif Filter[a][b+r] == 1:
                    #print("Right side in filter")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                aperture[y+a][x+b] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                    galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    #plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()                    
                    galaxy_counts = 0
                    counts_err = 0
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image , New_Filter                   
                
                else:
                    pixels = 0
                    #print ("No boundary conditions")
                    for x in np.arange(-r, r):
                        for y in np.arange(-r, r):
                            if x**2 + y**2 <= r**2:
                                #print ("index = ", [a+y, b+x])
                                galaxy_counts += image[y+a][x+b] - local_backg
                                aperture[a+y][b+x] = 1
                                new_image[y+a][x+b] = local_backg
                                New_Filter[y+a][x+b] = 1
                                pixels += 1
                    ##galaxy = aperture*image
                    #plt.imshow(aperture[a-100:a+100, b-100:b+100])
                    #plt.show()
                    ##plt.imshow(galaxy[a-100:a+100, b-100:b+100])
                    ##plt.show()
                    #plt.imshow(new_image[a-100:a+100, b-100:b+100])
                    #plt.show()
                    counts_err = backg_err/np.sqrt(pixels)
                    #print ("Number of pixels", pixels)
                    #print ("Local background = ", local_backg)
                    #print ("Galaxy Counts = ", galaxy_counts)
                    #print ("Counts Error = ", counts_err)
                    #print ("Radius = ", r)
                    return r, local_backg, backg_err, galaxy_counts, counts_err, new_image, New_Filter
                
def Algorithm(image_data, new_filename, interval = 0.25, Filter_Function = \
              Initial_Filter(image_data), thresh=3.5, lower = 25, \
              image_background = mean_background, image_stdev = \
              background_stdev, minimum_radius = 1.9946483113443867):
    """This is the algorithm used to identify, sort and catalogue galaxies and
    other bright objects in the image.
    
    INPUTS
    -------
        image_data = 2D array containing all of the pixel values in the image
        new_filename = Used to create a new set of files for this dataset.
        interval = Magnitude intervals in which the galaxies are sorted
        Filter_Function = The initial filter applied to the image, and the i
            image that is returned from it
        thresh = Number of standard deviations above the mean background at 
            which the algorithm stops
        lower = The lower radius of the annulus used to calculate the local 
            background of non-bleeding bright objects
        image_background = Mean value of the image background
        image_stdev = Standard deviation of background noise in the image
        minimum_radius = Minimum radius of a bright object that is required to
            classify it as a galaxy (as opposed to noise)
            
    OUTPUTS
    -------
        See documentation below for the output files that are generated from 
        this function."""
    
    
    
    #Applying the initial filter to the image:
    
    Filter = Filter_Function[1]
    Image = Filter_Function[0]
    
    Galaxy_Counts = np.array([])
    Galaxy_Errors = np.array([])
    Radii = np.array([])
    
    index = Locus(Image)[1]
    brightest_spot = Locus(Image)[0]
    background =  image_background
    iterations = 0
    print ("Brightest Spot = ", index, brightest_spot)
    print ("Background = ", background)
    print ("Theshold = ", thresh*image_stdev)
    
    
    while brightest_spot > image_background + thresh*image_stdev:
        index = Locus(Image)[1]
        #plt.imshow(Image[index[0]-100:index[0]+100, index[1]-100:index[1]+100])
        #plt.show()
        #plt.imshow(Filter[index[0]-100:index[0]+100, index[1]-100:index[1]+100])
        #plt.show()
        star_results = StarMask(index, Image, Filter)
        IsStar = star_results[0]
        galaxy_results = Galaxy(index, Image, Filter)
        radius = galaxy_results[0]
        counts = galaxy_results[3]
        counts_err = galaxy_results[4]
        #print ("IsStar = ", IsStar)
        if IsStar == True:
            print ("Star Masked")
            Image = star_results[1]
            Filter = star_results[2]
            brightest_spot = Locus(Image)[0]
            background = np.median(Image)
            #plt.imshow(Image[index[0]-100:index[0]+100, index[1]-100:index[1]+100])
            #plt.show()
            #plt.imshow(Filter[index[0]-100:index[0]+100, index[1]-100:index[1]+100])
            #plt.show()
            print ("Brightest Spot = ", index, brightest_spot)
            print ("Background = ", background)
            print ("Theshold = ", thresh*image_stdev)
        elif counts > 0 and radius > minimum_radius:
            print ("Galaxy Counted")
            Radii = np.append(Radii, radius)
            Galaxy_Counts = np.append(Galaxy_Counts, counts)
            Galaxy_Errors = np.append(Galaxy_Errors, counts_err)
            Image = galaxy_results[5]
            Filter = galaxy_results[6]
            brightest_spot = Locus(Image)[0]
            background = np.median(Image)
            #plt.imshow(Image[index[0]-100:index[0]+100, index[1]-100:index[1]+100])
            #plt.show()
            #plt.imshow(Filter[index[0]-100:index[0]+100, index[1]-100:index[1]+100])
            #plt.show()
            print ("Brightest Spot = ", index, brightest_spot)
            print ("Background = ", background)
            print ("Theshold = ", thresh*image_stdev)
            if radius >= lower:
                raise ValueError\
                ("Galaxy too big. Increase 'lower' annulus parameter.")
        else:
            print ("Galaxy NOT Counted")
            Image = galaxy_results[5]
            Filter = galaxy_results[6]
            brightest_spot = Locus(Image)[0]
            background = np.median(Image)
            #plt.imshow(Image[index[0]-100:index[0]+100, index[1]-100:index[1]+100])
            #plt.show()
            #plt.imshow(Filter[index[0]-100:index[0]+100, index[1]-100:index[1]+100])
            #plt.show()
            print ("Brightest Spot = ", index, brightest_spot)
            print ("Background = ", background)
            print ("Theshold = ", thresh*image_stdev)
            if radius >= lower:
                raise ValueError\
                ("Galaxy too big. Increase 'lower' annulus parameter.")
        iterations += 1

    print ("Number of galaxies detected = ", len(Galaxy_Counts))
    print ("Number of iterations = ", iterations)
    
    #Converting galaxy counts into magnitudes
    Galaxy_Magnitudes = 2.530E+01 - 2.5* np.log10(Galaxy_Counts)
    Magnitude_Errors = np.sqrt((2.000E-02)**2 + (2.5*Galaxy_Errors/(Galaxy_Counts*np.log(10))**2))
    
    #Sorting the galaxy magnitudes into bins    
    Magnitudes = np.arange(min(Galaxy_Magnitudes), max(Galaxy_Magnitudes), interval)
    Number_of_Galaxies = np.zeros(len(Magnitudes))
    Number_Error = np.zeros(len(Magnitudes))
    
    i = 0
    while i < len(Magnitudes):
        Number_of_Galaxies[i] = sum(p < Magnitudes[i] + interval and p >= \
                          Magnitudes[i] for p 
              in Galaxy_Magnitudes) + sum(Number_of_Galaxies[0:i])
        Number_Error[i] = 1/np.sqrt(Number_of_Galaxies[i])
        i += 1
    
    Log_Number = np.log10(Number_of_Galaxies)
    Log_Number_Error = Number_Error/(Number_of_Galaxies*np.log(10))
    
    plt.clf()
    plt.plot(Magnitudes, Log_Number, 'o')
    plt.errorbar(Magnitudes, Log_Number, yerr = Log_Number_Error, fmt = '.k')
    plt.xlabel('Magnitude')
    plt.ylabel('Log Number of Galaxies')
    plt.title('Cumulative Number of Galaxies at Increasing Magnitudes')
    plt.show()
    
    plt.clf()
    plt.plot(Galaxy_Magnitudes, Radii, 'o')
    plt.xlabel('Magnitude')
    plt.ylabel('Radius (Number of pixels)')
    plt.title('Galaxy Radius and Brightness')
    plt.show()
    
    #Saving the final image and filter in an .fits file
    final_image = fits.PrimaryHDU(Image)
    final_image.writeto(f'Final Image ({new_filename}).fits')
    appended_filter = fits.PrimaryHDU(Filter)
    appended_filter.writeto(f'Final Filter({new_filename}).fits')
    
    #Saving the data in a .npy file
    Binned_Data = [Magnitudes, Number_of_Galaxies, Number_Error]
    np.save(f'Binned Data ({new_filename})', Binned_Data)
    Galaxy_Data = [Galaxy_Magnitudes, Magnitude_Errors, Radii]
    np.save(f'Galaxy Data ({new_filename})', Galaxy_Data)
    Raw_Data = [Galaxy_Counts, Galaxy_Errors, Radii]
    np.save(f'Raw Galaxy Data ({new_filename})', Raw_Data)
    
    #Saving the data in a .csv file
    np.savetxt(f'Binned_Data({new_filename}).csv', [p for p in zip(Magnitudes, \
                                                 Number_of_Galaxies)], \
    delimiter = ',', fmt = '%s' )
    np.savetxt(f'Galaxy_Data({new_filename}).csv', [p for p in zip(Galaxy_Magnitudes, \
                                                  Magnitude_Errors, Radii)], 
    delimiter = ',', fmt = '%s')
    np.savetxt(f'Raw_Data({new_filename}).csv', [p for p in zip(Galaxy_Counts, \
                                                  Galaxy_Errors, Radii)], 
    delimiter = ',', fmt = '%s')
    
    return [Magnitudes, Number_of_Galaxies]

def Analysis(data_file, interval = 0.25):
    """This function converts the catalogued galaxy magnitudes from Algorithm
    and plots the logarithm of cumulative galaxy counts with respect to
    magnitude. It also plots galaxy size with respect to magnitude.
    
    INPUTS
    ------
    data_file = new_filename from Algorithm
    interval = Magnitude interval in which the galaxy data is binned.
    
    OUTPUTS
    -------
    See documentation below for the output .csv file containing the data for
    the Log(Counts) vs Magnitude plots."""
    
    galaxy_data = np.load(f'Galaxy Data ({data_file}).npy')
    raw_galaxy_data = np.load(f'Raw Galaxy Data ({data_file}).npy')
    
    Galaxy_Magnitudes = galaxy_data[0]
    Galaxy_Counts = raw_galaxy_data[0]
    Galaxy_Errors = raw_galaxy_data[1]
    Magnitude_Errors = np.sqrt((2.000E-02)**2 + (2.5*Galaxy_Errors/(Galaxy_Counts*np.log(10))**2))
    Radii = galaxy_data[2]
    
    Magnitudes = np.arange(min(Galaxy_Magnitudes), max(Galaxy_Magnitudes), 
                           interval)
    mean_radius = np.zeros(len(Magnitudes))
    radius_error = np.zeros(len(Magnitudes))
    number_of_galaxies = np.zeros(len(Magnitudes))
    indices = np.array([], dtype = int)
    interval_errors = np.array([])
    cumulative_magnitude_error = np.zeros(len(Magnitudes))
    interval_radii = np.array([])
    cumulative_count_error = np.zeros(len(Magnitudes))
    cumulative_number_of_galaxies = np.zeros(len(Magnitudes))
    mean_magnitude_error = np.zeros(len(Magnitudes))
    catalogued_errors = np.array([])
    
    i = 0
    while i < len(Magnitudes):
        number_of_galaxies[i] = sum(p < Magnitudes[i] + interval and p >= \
                          Magnitudes[i] for p 
              in Galaxy_Magnitudes)
        cumulative_number_of_galaxies[i] = number_of_galaxies[i] + sum(number_of_galaxies[0:i])
        print ("Number of galaxies in interval = ", number_of_galaxies[i])
        print ("i, galaxy count: ", i, cumulative_number_of_galaxies[i])
        for p in Galaxy_Magnitudes:
                if p < Magnitudes[i] + 0.5 and p >= Magnitudes[i]:
                    indices = np.append(indices, 
                                        Galaxy_Magnitudes.tolist().index(p))
        for j in indices:
            interval_errors = np.append(interval_errors, Magnitude_Errors[j])
            interval_radii = np.append(interval_radii, Radii[j])
            catalogued_errors = np.append(catalogued_errors, Magnitude_Errors[j])
        mean_magnitude_error[i] = np.mean(interval_errors)
        cumulative_magnitude_error[i] = \
        np.sqrt(sum(catalogued_errors**2)/cumulative_number_of_galaxies[i]**2)
        cumulative_count_error[i] = 1/np.sqrt(cumulative_number_of_galaxies[i])
        mean_radius[i] = np.mean(interval_radii)
        radius_error[i] = np.std(interval_radii)
        interval_errors = np.array([])
        interval_radii = np.array([])
        i += 1
        
    Log_Number = np.log10(cumulative_number_of_galaxies)
    Log_Number_Error = cumulative_count_error/(
            cumulative_number_of_galaxies*np.log(10))
    
    
    plt.clf()
    plt.plot(Magnitudes, mean_radius, 'o')
    plt.errorbar(Magnitudes, mean_radius, yerr = radius_error, fmt = '.k')
    plt.xlabel('Magnitudes')
    plt.ylabel('Mean Radius')
    plt.title('Galaxy Size at Increasing Magnitudes')
    plt.show()
    
    plt.clf()
    plt.plot(Magnitudes, Log_Number, 'o')
    plt.errorbar(Magnitudes, Log_Number, yerr = Log_Number_Error, xerr = cumulative_magnitude_error, fmt = '.k')
    plt.xlabel('Magnitudes')
    plt.ylabel('Log(Cumulative Number of Galaxies)')
    plt.title('Logarithm of Cumulative Number of Galaxies at Increasing Magnitudes')
    plt.show()
    
    
    print ('Maximum Radius = ', max(mean_radius))
    print ('Minimum Radius = ', np.mean(mean_radius[len(mean_radius)-15:-1]))
    print ('Minimum Radius Standard Deviation = ', radius_error[-1])
    np.savetxt(f'Log_Count_vs_Magnitude({data_file}).csv', [p for p in \
               zip(Magnitudes, Log_Number, Log_Number_Error, \
                   cumulative_magnitude_error)], delimiter = ',', fmt = '%s' )
    
    
    