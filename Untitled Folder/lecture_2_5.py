import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loads an image and converts it to grayscale
def loadImage(filename, show):
	img = cv2.imread(filename)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if show>0:
		cv2.namedWindow("gray scaled image")
		cv2.imshow("gray scaled image",img_gray)
		cv2.waitKey(25)
	return img_gray

# Takes an image and returns a histogram
def histImg(img):
	img=np.array(img)
	if not img.ndim == 2:
		"You need to pass a gray scale image"
	else:
		H, W = np.shape(img)
		hist_values=np.zeros(256)
		for r in range(H):
			for c in range(W):
				value=img[r][c]+1
				hist_values[value]+=1
		plt.plot(hist_values)
		plt.xlabel("Pixel value")
		plt.ylabel("Number of pixels")
		plt.show()
		return hist_values

# This function uses in built functions to calculate the histograms
# If the second argument, func, is greater than 0 then the matplotlib function is used
# Otherwise the cv2 in built function is used
def histImgFuncs(img,func):
	img=np.array(img)
	if not img.ndim==2:
		print "You need to pass a gray scale image"
	elif func>0:
		plt.hist(img.ravel(),256,[0,256])
		plt.show()
		# A numpy function can also be used that is similar to the above:
		# hist, bins = np.histogram(img.ravel(),256,[0,256])
	else:
		img_hist=cv2.calcHist([img],[0],None,[256],[0,256])
		plt.plot(img_hist)
		plt.show()
				