import numpy as np
import cv2

# Returns a binary image using a threshold value
# Note: there is also an in built function cv2.threshold
def threshold(img, thresh_value):
	img = np.array(img)
	height,width = np.shape(img)
	output=np.zeros([height,width])
	for row in range(height):
		for col in range(width):
			if img[row][col]>thresh_value:
				output[row][col] = 1
	return output

# Create a gaussian smoothing window and applies it to histogram
def threshHistSmooth(img_hist, n, w, sigma):
    nn = int((n-1)/2)
    gauss = np.asarray([(w/n)*x**2 for x in range(-nn,nn+1)], dtype=float)
    gauss = np.exp(-gauss/(2*sigma**2))
    the_filter = gauss/sum(gauss)
    hist_convolve = np.convolve(np.ravel(img_hist), the_filter)
    return hist_convolve

# Finds a threshold using the trough between peaks
def peakPick(img):
    img_hist=cv2.calcHist([img],[0],None,[256],[0,256])
    img_hist=threshHistSmooth(img_hist, 50, 5, 1)
    
    # Find peak in histogram
    peak = np.argmax(img_hist)
    # Find peak in the darker side of the above peak value
    peak_darker = 1
    for i in range(1,peak):
    	if img_hist[i-1] < img_hist[i] and img_hist[i] >= img_hist[i+1] and img_hist[i]>img_hist[peak_darker]:
    		peak_darker=i
    # print peak_darker
    # Find the deepest valley bewteen the two peaks
    # This will be our threshold value
    thresh = peak_darker+1
    for i in range(peak_darker+1,peak):
    	if img_hist[i-1] > img_hist[i] and img_hist[i] <= img_hist[i+1] and img_hist[i] < img_hist[thresh]:
    		thresh = i
    return thresh
