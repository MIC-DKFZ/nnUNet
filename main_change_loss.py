from sklearn.model_selection import train_test_split
import cv2, torch, torchvision
import exploratory_data_analysis, util.util as util, models.unet.multi_class_unet as multi_class_unet
import glob, pylab, pydicom, pandas as pd
import numpy as np
import tensorflow as tf

#from PIL import Image


 
#model.compile(optimizer=Adam(learning_rate=0.001), loss=CustomAccuracy(), metrics=['mae', 'mse'])

def main():
    HU_PixelData = []
    dcmDataSet, dcmPixelData = util.data_loader("Cleaned_image")
    metadata = util.dcm_metadata(dcmDataSet)

    #Compiling the model
    model = multi_class_unet.multi_unet_model()
    #model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    ###
    #  To be refactored
    dcmFilePath = "Cleaned_image/*"
    dcmFileList = glob.glob(dcmFilePath)
    temp = [("Cleaned_mask/"+ x.split("/")[1]) for x in dcmFileList]
    maskFileList = [(r+".tag") for r in temp]

    #reading the metadata from dicom image
    dcmData = [pydicom.read_file(x) for x in dcmFileList]
    #reading the pixel data from the dicom image
    dcmPixelData = [x.pixel_array.reshape(512,512,1) for x in dcmData]

    #Reading the masked dicom image
    mask= []
    
    for i in range(len(maskFileList)):
        mask.append(util.maskedImageReader(dcmData[i], maskFileList[i]))
        #mask[i] = mask[i].reshape(512, 512, 1)
        mask[i] = tf.one_hot(mask[i].reshape(512, 512),4)

    ##Test-Train Data split
    #x_train, x_test, y_train, y_test = train_test_split(dcmPixelData, mask, test_size=0.237037037, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(dcmPixelData, mask, test_size=0.3)

    ## Fit Model 

    
    
         
    model.compile(optimizer='adam',
              loss="BinaryFocalCrossentropy",
              metrics=['accuracy'])
        
                       
    history = model.fit(x=np.array(x_train), y=np.array(y_train), epochs=5, batch_size=2)
    #print(model.summary())
    
    
   
    
    #Predict
    result = model.predict(np.array(x_test))
    predicted_indices = tf.argmax(result, 1)
    #predicted_class = tf.gather(TARGET_LABELS, predicted_indices)
    #print(predicted_indices)    
     #Generating heatmap
    #for i in range(len(y_train)):
    #model.layers[-1].activation = None
    #heatmap = []
    #heatmap = util.make_gradcam_heatmap(np.expand_dims(np.array(x_test[0]), axis=0), model, model.layers[-4].name, maskClass=4)
    #util.saveImage(heatmap, "Heatmap", cmap)
    #util.display_gradcam(np.array(x_test[0]), heatmap, 0.3, "abc")
    #for i in range(5):
    #    util.display_gradcam(np.array(x_test[i]), heatmap[i], 0.3, i)
    
'''
    for i in range(len(y_train)):
        mask_image = np.zeros((4, 512, 512))
        with open(y_train[i], 'rb') as f:
            mask_train.append(util.maskedImageReader(dcmData_train[i], y_train[i]))
            mask_array = util.transform_to_hu(dcmData_train[i], mask_train[i])
            px_array = util.transform_to_hu(dcmData_train[i], dcmData_train[i].pixel_array)
            mask_image[0] = np.where((px_array > -29) & (px_array < 150), mask_array, 0)
            mask_image[1] = (np.where((px_array > -150) & (px_array < -50), mask_array, 0))
            mask_image[2] = (np.where((px_array > -190) & (px_array < -30), mask_array, 0))
            mask_image[3] = (np.where((px_array > -190) & (px_array < -30), mask_array, 0))
            mask_image = np.stack(mask_image, axis=-1)
            x = dcmPixelData_train[i].reshape(1, 512, 512)
            y = mask_image.reshape(1, 512, 512, 4)
            model.fit(x,y)

'''

if __name__ == "__main__": 
    main()