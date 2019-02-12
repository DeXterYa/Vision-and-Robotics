import cv2
import numpy as np
import matplotlib.pyplot as plt
import lecture_2_5 as l25
import lecture_3_2 as l32
import lecture_4_2 as l42

# This code uses support vector machine on the flat part images
# There are three classes; part1, part2, and part3
# These training samples are stored in the folder train_images
# There s a test image in the directory test_image which is part2
#
# This code is under the assumption that the images are black and 
# white

class classifer():

	# classes should be a 2D array where
	# the first value is the name of the class, 
	# and the second is the number of training
	# instances
	def __init__(self, classes, no_dim=3):
		self.svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=10, gamma=5.383 ) #2.67
		self.classes=classes
		self.dim=no_dim

		self.samples=np.array([[]], dtype=np.float32)
		self.samples_labels=np.array([], dtype=np.int)
		self.svm = cv2.ml.SVM_create()
		self.svm.setType(cv2.ml.SVM_C_SVC)
		self.svm.setKernel(cv2.ml.SVM_LINEAR)
		self.svm.setTermCriteria((cv2.	TERM_CRITERIA_COUNT, 100, 1.e-6))

	def addTrainSamples(self, folder):
		kernel = np.ones((5,5),np.uint8)
		class_key=0
		for c in self.classes:
			for i in range(1,c[1]+1):
				filename=folder+'/'+c[0]+str(i)+'.jpg'
				print(filename)      
				img=l25.loadImage(filename,0)
				thresh=l32.peakPick(img)
				img_bin=abs(l32.threshold(img, 127)-1) # Given this set of data we need to flip the values
# 				plt.imshow(img_bin, cmap='hot', interpolation='nearest')
# 				plt.show()
				img_props=np.array([l42.getproperties(img_bin)], dtype=np.float32)
				if self.samples.size==0:
					self.samples=img_props
				else:
					self.samples=np.append(self.samples,img_props, axis=0)
				self.samples_labels=np.append(self.samples_labels, np.array([class_key], dtype=np.int))
			class_key+=1
		return

	def train(self, model="", load=False):
		if load:
			try:
				self.svm = self.svm.load(model)
			except:
				print "Provide a valid xml file."
		else:
			self.svm.train(self.samples, cv2.ml.ROW_SAMPLE, self.samples_labels)
			if model!="":
				try:
					self.svm.save(model)
				except:
					print "The filename must be valid."
		return

	def classify(self, filename):
		img = l25.loadImage(filename,0)
		kernel = np.ones((5,5),np.uint8)
		thresh=l32.peakPick(img)
		img_bin=abs(l32.threshold(img, 127)-1) # Given this set of data we need to flip the values
		img_props=np.array([l42.getproperties(img_bin)], dtype=np.float32)
		prediction=self.svm.predict(img_props)
		return prediction

	def img_classify(self, img):
		kernel = np.ones((5,5),np.uint8)
		thresh=l32.peakPick(img)
		img_bin=abs(l32.threshold(img, 127)-1) # Given this set of data we need to flip the values
		img_props=np.array([l42.getproperties(img_bin)], dtype=np.float32)
		prediction=self.svm.predict(img_props)
		return prediction


