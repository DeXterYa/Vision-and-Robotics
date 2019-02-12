import cv2
import numpy as np

# Returns list of moments from a binary image
# Currently only returning compactness, ci1 and ci2
# Note: there is an inbuilt function called cv2.moments() that does this
def getproperties(img):
	img = np.array(img)
	area = sum(img.ravel())

	perim = bwperim(img)

	compactness = perim*perim/(4.0*np.pi*area)

	c11 = complexmoment(img,1,1) / np.power(area,2)
	c20 = complexmoment(img,2,0) / np.power(area,2)
	c30 = complexmoment(img,3,0) / np.power(area,2.5)
	c21 = complexmoment(img,2,1) / np.power(area,2.5)
	c12 = complexmoment(img,1,2) / np.power(area,2.5)

	ci1 = np.real(c11)
	ci2 = np.real(1000*c21*c12)
	tmp = c20*c12*c12
	ci3 = 10000*np.real(tmp)
	ci4 = 10000*np.imag(tmp)
	tmp = c30*c12*c12*c12
	ci5 = 1000000*np.real(tmp)
	ci6 = 1000000*np.imag(tmp)

	return [compactness,ci1,ci2,ci3,ci4,ci5,ci6]

# Returns the perimeter of the binary blob.
# Note: There are in built functions from opencv that calculates this using the moments
def bwperim(img):
	img=np.array(img)
	height,width = np.shape(img)
	perim_img=np.zeros([height,width])
	for row in range(height):
		for col in range(width):
			if row==0 or row==height-1 or col==0 or col==width-1:
				if img[row][col]==1:
					perim_img[row][col]=1
			else:
				if img[row][col]==1 and (img[row][col+1]==0 or img[row+1][col+1]==0 or img[row+1][col]==0 or img[row+1][col-1]==0 or img[row][col-1]==0 or img[row-1][col-1]==0 or img[row-1][col]==0 or img[row-1][col+1]==0):
					perim_img[row][col]=1
	# Uncomment the next two lines if you want to see the perimeter
	# cv2.imshow("perim image",perim_img)
	# cv2.waitKey(5)
	return sum(perim_img.ravel())

# Returns a given complex moment
def complexmoment(img, u, v):
	img = np.array(img)
	indices = np.argwhere(img>0)
	centre = np.mean(indices,0)

	momlist = np.zeros(np.shape(indices)[0],dtype=complex)
	for i in indices:
		c1 = i[0]-centre[0]+(i[1]-centre[1])*1j
		c2 = i[0]-centre[0]+(centre[1]-i[1])*1j
		momlist[i]=np.power(c1,u)*np.power(c2,v)

	return sum(momlist)