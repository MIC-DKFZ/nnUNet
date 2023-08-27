"""
Created on Sun Apr 9 2023

@author: nsingla
"""

import pylab, glob, pydicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from keras.metrics import MeanIoU


def data_loader(folderpath):
    #Reading all the dcm images from the directory
    dcmFilePath = folderpath + "/*"
    dcmFileList = glob.glob(dcmFilePath)

    #should there be a requirement to get the name of the indi files. 
    #dcm_file_list = [x.split("/")[1] for x in fileList]

    #reading the metadata from dicom image
    dcmData = [pydicom.read_file(x) for x in dcmFileList]

    #reading the pixel data from the dicom image
    dcmPixelData = [x.pixel_array for x in dcmData]
    return dcmData, dcmPixelData

def maskedImageReader(dcmdata, maskImage):
    with open(maskImage, 'rb') as f:
        numpy_data = np.fromfile(f, dtype=np.uint8)
        to_trim = numpy_data.shape[0] - dcmdata.pixel_array.flatten().shape[0]
        trimmed = numpy_data[to_trim:, ...]
        masked_image = np.reshape(trimmed, dcmdata.pixel_array.shape)
    return masked_image

def transform_to_hu(medical_image, pixel_array):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = pixel_array * slope + intercept

    return hu_image


def dcm_metadata(dcmData):
    metadata = []
    tags = ['ContrastBolusAgent', 'ContrastBolusRoute', 'PatientAge', 'PatientID', 'PatientSex', 'SliceThickness', 'SliceLocation']
    for dd in dcmData:
        metadir = {}
        for tag in tags:
            #metadir[tag] = dd.data_element(tag).value
            metadir[tag] = dd.get(tag)
        metadata.append(metadir)
    metadata = pd.DataFrame(metadata)
    metadata["PatientAge"] = metadata["PatientAge"].apply(lambda x: int(x[1:3]))
    return metadata


def saveImage(img, name, map = plt.cm.gist_gray):
    #pylab.imshow(img, cmap=map)
    #pylab.axis('off')
    filePath = "EDA_Result/" + name + ".png"
    #pylab.savefig(filePath)
    plt.imshow(img, cmap=map)
    plt.savefig(filePath)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, maskClass, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    #maskClass = maskClass-1
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        #if pred_index is None:
#            print("preds.shape: ", preds.shape)
            #pred_index = tf.argmax(preds[0, :, :, maskClass])
#            print("pred_index: ", pred_index)
        #class_channel = preds[:, pred_index]
        class_channel = tf.reduce_sum(preds[0, :, :, maskClass], axis=0)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.nn.relu(heatmap)
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Define a function to create a superimposed visualization
def display_gradcam(img, heatmap, alpha=0.3, name = "SuperImposed_image"):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap('jet')

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Display superimpose result
    plt.imshow(superimposed_img)
    plt.title('Superimpose')
    plt.axis('off')
    cmap = plt.cm.get_cmap("gnuplot2")
    saveImage(superimposed_img, name, cmap)

def saveOneHotEncoded(result):

    # Convert one-hot encoded to label image
    label_image = np.argmax(result, axis=-1)

    # Define class color map
    colors = {
        0: [1, 0, 0],  # class 0: red
        1: [0, 1, 0],  # class 1: green
        2: [0, 0, 1],  # class 2: blue
        3: [1, 1, 0]   # class 3: yellow
    }

    # Convert label image to color image
    color_image = np.zeros((*label_image.shape, 3))
    for label, color in colors.items():
        mask = label_image == label
        color_image[mask] = color

    # Display color image
    plt.imshow(color_image)
    plt.savefig("EDA_result/Result_0")

def evaluation_metric(predictions, true_label):

    classes = [0,1,2,5,7]
    ##Calculate pixel accuracy in the result
    pixel_accuracy = []
    accuracy = []
    total_pixel = np.prod([predictions.shape[1], predictions.shape[2]])

    for i in range(len(predictions)):
        prediction_label = np.argmax(predictions[i], axis=-1)
        all_zeros_prediction = np.all(predictions[i]==0, axis=-1)
        prediction_label[all_zeros_prediction] = -1
        label = np.argmax(true_label[i], axis=-1)
        all_zeros_label = np.all(true_label[i]==0, axis=-1)
        label[all_zeros_label] = -2.
        matching_pixel = np.sum(prediction_label == label)
        pixel_accuracy.append(matching_pixel/total_pixel)

    ## Calculate class-wise accuracy
        class_accuracy = {}
        for j in range(len(classes)-1):
            mask = label == j+1
            class_pixel = np.sum(mask)
            class_matching_pixel = np.sum(np.logical_and(mask, prediction_label == j+1))
            class_accuracy[j] = class_matching_pixel/class_pixel
        accuracy.append(class_accuracy)
    return pixel_accuracy, accuracy

def hist(img, name):
    filePath = "EDA_Result/" + name + ".png"
    histogram, bin_edges = np.histogram(img, bins=256)
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    #plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram)
    plt.savefig(filePath)

def calculateIOU(y_test, result, n_classes):

    IoU_keras = MeanIoU(num_classes = n_classes)
    IoU_keras.update_state(y_test[:,:,:,0], np.argmax(result, axis=-1))
    print("Mean IoU = ", IoU_keras.result().numpy())


    # Calculate IoU for each class
    values = np.array(IoU_keras.total_cm)
    print(values)

    class_IoU = []
    for i in range(n_classes):
        print(i)
        print(values[i,i])
        class_IoU.append(values[i, i]/(np.sum(values[i, :]) + np.sum(values[:, i]) - values[i,i]))
        print("IoU for class ", i ," = ", class_IoU[i])
    return class_IoU

def dice_coefficient():
    # DSC = 2TP/(2TP + FP + FN)
    return None
