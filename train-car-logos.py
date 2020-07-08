from sklearn.neighbors import KNeighborsClassifier
import joblib                                    
from skimage import feature
import cv2
import os
import time 
from progress.bar import IncrementalBar

inputPath = 'E:\Zuq\Studying\DIP\Car-Logos-Recognition\test_images/'
datasetPath = 'E:\Zuq\Studying\DIP\Car-Logos-Recognition\car_logos/'
modelPath = 'E:\Zuq\Studying\DIP\Car-Logos-Detection/model/'
modelName = 'carLogoModel.pkl'

imageSize = (200,100)

print('[INFO] Extracting features...')
data=[]
labels=[]

def trainModel():
    print ('[INFO] Initiating Training Model')
    print ('[INFO] Image Width: ' + str(imageSize[0]))
    print ('[INFO] Image Height: ' + str(imageSize[1]))
    
    initialTime  = time.time()           #Initial Time 

    # Count the number of different cars logo for the progress bar 
    barCount = 0 
    for logos in os.listdir(datasetPath):
        if os.path.isdir(datasetPath+logos):
            barCount +=1

    bar = IncrementalBar('Training', max = barCount)

    for logos in os.listdir(datasetPath):
        if os.path.isdir(datasetPath+logos):
            bar.next()
            label = logos 
            for imageName in os.listdir(datasetPath+logos):

                # Load image from the path
                imagePath = datasetPath+logos+ '/' + imageName
                print(imagePath);
                image=cv2.imread(imagePath)
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                canny = cv2.Canny(gray,100,255)
                
                contours, hierarchy=cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                maxCnt=max(contours,key=cv2.contourArea)

                # Extract the logo of the car and resize it to a canonical width  and height
                x,y,w,h=cv2.boundingRect(maxCnt)
                logo = gray[y:y + h, x:x + w]
                logo = cv2.resize(logo, imageSize)
                H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                    cells_per_block=(2, 2), transform_sqrt=True)

                # Update the data and labels
                data.append(H)
                labels.append(label)
    bar.finish()             

    # Train the nearest neighbors classifier
    print ("[INFO] Training kNN classifier")
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(data, labels)

    print("[INFO] Saving ML Model")
    joblib.dump(model,modelPath + modelName)
    print('[INFO] ML Model Trained, Time Taken: ' + str(round((time.time() - initialTime),3)) + ' sec')
    print('[INFO] ML Model Name: ' + str(modelName))
    modelSize = round(os.stat(modelPath+modelName).st_size / (1024 * 1024),3)
    print('[INFO] ML Model size: ' + str(modelSize) + ' MB')
    return model

trainModel()