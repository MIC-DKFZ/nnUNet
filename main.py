from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import exploratory_data_analysis, util.util as util, models.unet.multi_class_unet as multi_class_unet
import glob, pylab, pydicom, pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize
from keras.utils import to_categorical
from sklearn.utils import class_weight
from keras.metrics import MeanIoU
import plotly.express as px

##
#Initialisation
n_classes = 5
class_label = {1: 'SM', 2: 'IMAT', 3: 'VAT', 4: 'SAT'}

###########################3########################  Load DataSet #################################################

dcmFilePath = "Cleaned_image/*"
dcmFileList = glob.glob(dcmFilePath)
temp = [("Cleaned_mask/"+ x.split("/")[1]) for x in dcmFileList]
maskFileList = [(r+".tag") for r in temp]
dcmData = [pydicom.read_file(x) for x in dcmFileList]
dcmPixelData = [x.pixel_array for x in dcmData]
dcmPixelData = np.array(dcmPixelData)

###################################################################################################################


####################################################  Load Mask ###################################################
mask = []
for i in range(len(maskFileList)):
        mask.append(util.maskedImageReader(dcmData[i], maskFileList[i]))

mask = np.array(mask)
mask = np.where(mask == 14, 0, mask)
labelencoder = LabelEncoder()
n,h,w = mask.shape
mask_reshaped = mask.flatten()
mask_reshaped_encoded = labelencoder.fit_transform(mask_reshaped)
mask_encoded_original_shape = mask_reshaped_encoded.reshape(n,h,w)

###################################################################################################################


###################################################  Process Data #################################################

dcmPixelData = np.expand_dims(dcmPixelData, axis=3)
dcmPixelData = normalize(dcmPixelData, axis=1)

mask_input = np.expand_dims(mask_encoded_original_shape, axis = 3)

#Picking 10% for testing and remaining for training

X1, X_test, y1, y_test = train_test_split(dcmPixelData, mask_input, test_size = 0.10, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

#print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

train_mask_cat = to_categorical(y_train, num_classes=n_classes)

y_train_cat = train_mask_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_mask_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_mask_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

###################################################################################################################

############################################### Balancing class weight ############################################

#class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(mask_reshaped_encoded),
#                                                mask_reshaped_encoded)

#print("Class weights are.....:", class_weights)

###################################################################################################################


################################################# Metadata analysis ###############################################

metadata = util.dcm_metadata(dcmData)

exploratory_data_analysis.agedistribution(metadata, "AgeDistribution")
exploratory_data_analysis.genderDistribution(metadata, "GenderDistribution")

###################################################################################################################


############################################### Compile and Train Model ###########################################

model = multi_class_unet.multi_unet_model()

model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

model.summary()

callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir = "Logs")]

history = model.fit(X_train, y_train_cat, 
                    callbacks= callbacks, 
                    batch_size = 8, 
                    epochs=200, 
                    validation_data=(X_test, y_test_cat), 
                    shuffle=False)

_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

##################################################### Prediction ###################################################

result = model.predict(np.array(X_test))

pixelAccuracy, Accuracy = util.evaluation_metric(result[0:1], y_test_cat[0:1])
print("Pixel Accuracy: ", pixelAccuracy)
#print("Class-wise Accuracy: ", Accuracy)
for i in range(1, n_classes):
        print("Class wise accuracy for ", class_label[i], "=", Accuracy[0][i-1])
print()
###################################################################################################################


################################################## Generate Heatmap ###############################################

for i in range(5):
        maskClass = i+1
        heatmap = util.make_gradcam_heatmap(np.expand_dims(X_test[0], axis=0), model, model.layers[-2].name, maskClass=i)
        cax = plt.matshow(heatmap)
        plt.colorbar(cax)
        imageName = "EDA_result/plt_Class_" + str(maskClass) + ".png"
        plt.savefig(imageName)
        filename = "abc" + str(i)
        util.display_gradcam(np.array(X_test[0]), heatmap, 0.5, filename)

###################################################################################################################


################################################### Calculate IoU #################################################

IoU_keras = MeanIoU(num_classes = n_classes)
IoU_keras.update_state(y_test[:,:,:,:], np.argmax(result, axis=-1))
print("Mean IoU = ", IoU_keras.result().numpy())
print()

# Calculate IoU for each class
values = np.array(IoU_keras.total_cm)

class_IoU = []
for i in range(1, n_classes):
        class_IoU.append(values[i, i]/(np.sum(values[i, :]) + np.sum(values[:, i]) - values[i,i]))
        print("IoU for class ", class_label[i] ," = ", class_IoU[i-1])
print()
#class_IoU = util.calculateIOU(y_test, result, n_classes)
###################################################################################################################


######################################## Calculate Sørensen–Dice coefficient ######################################
iou = np.array(class_IoU)
dice_coefficient = (2*iou)/(1+iou)
for i in range(1, n_classes):
        print("Dice Coefficient for ", class_label[i], "=", dice_coefficient[i-1])
print()
###################################################################################################################