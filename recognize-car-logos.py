import joblib                                    
from skimage import feature
from skimage import exposure
import cv2
import os
import time 

#Set path for test folder and model

inputPath='.../test_images/'
modelPath = '.../model/'
testImageName='test1.jpg'
modelName = 'carLogoModel.pkl'

imageSize = (200,100)

#Load trained model

def LoadTrainModel():
    print ('[INFO] Loading ML Model')
    initialTime = time.time()
    model=joblib.load(modelPath+modelName)
    print('[INFO] ML Model Loaded ,Time Taken: ' + str(round((time.time() - initialTime),3)) + ' sec')
    print('[INFO] ML Model Name: ' + str(modelName))
    modelSize = round(os.stat(modelPath+modelName).st_size / (1024 * 1024),3)
    print('[INFO] ML Model size: ' + str(modelSize) + ' MB')
    return model


def regconizeLogo():
	model = LoadTrainModel()
	'''Now loading the test image and predicting the results'''

	#Load test logo: convert to gray, resize to resolution 200x100 and find HOG of test logo
	
	image=cv2.imread(inputPath+testImageName)
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray=cv2.resize(gray, imageSize) 

	(H, hogImage)=feature.hog(gray,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),
	              transform_sqrt=True,visualize=True)
	pred = model.predict(H.reshape(1,-1))[0]
	print ('[INFO] Pridicted Result is: '+str(pred))

        #Show HOG of test logo
	
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
	cv2.imshow("HOG Image", hogImage)

        #Show test logo with predicted label
	
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
	            (0, 0, 255), 2)
	cv2.imshow("Test Image", image)


regconizeLogo()
cv2.waitKey(0)
